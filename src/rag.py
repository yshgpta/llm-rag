"""LangChain RAG pipeline backed by Gemini embeddings and Chroma."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .config import AppConfig
from .document_loader import LoadedDocument
from .monitoring import callback_config, start_child_observation, traced_observation


class RetryableEmbeddings:
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings, *, attempts: int = 3, delay_seconds: float = 1.0):
        self.embeddings = embeddings
        self.attempts = attempts
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self._with_retry(lambda: self.embeddings.embed_documents(texts, **kwargs))

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self._with_retry(lambda: self.embeddings.embed_query(text, **kwargs))

    def _with_retry(self, operation):
        last_error = None
        for attempt in range(1, self.attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc) or attempt == self.attempts:
                    break
                time.sleep(self.delay_seconds * attempt)
        raise RuntimeError(
            "Gemini embedding service failed while searching the document. "
            "This is usually temporary; please retry in a moment."
        ) from last_error


@dataclass
class RagAnswer:
    answer: str
    source_documents: list[Document]


@dataclass
class SimpleRagChain:
    config: AppConfig
    vector_store: Chroma
    documents: list[Document]
    callback: Any | None = None


def split_document(loaded: LoadedDocument, *, chunk_size: int = 1000, chunk_overlap: int = 180) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs = [Document(page_content=loaded.text, metadata={"source": loaded.filename, "type": loaded.source_type})]
    return splitter.split_documents(docs)


def collection_name_for(filename: str, text: str) -> str:
    digest = hashlib.sha256(f"{filename}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def build_embeddings(config: AppConfig) -> RetryableEmbeddings:
    if not config.google_api_key:
        raise ValueError("Google API key is required.")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.embedding_model, google_api_key=config.google_api_key)
    return RetryableEmbeddings(embeddings)


def build_llm(config: AppConfig) -> ChatGoogleGenerativeAI:
    if not config.google_api_key:
        raise ValueError("Google API key is required.")
    return ChatGoogleGenerativeAI(
        model=config.chat_model,
        google_api_key=config.google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )


def create_vector_store(
    loaded: LoadedDocument,
    config: AppConfig,
    *,
    embeddings: RetryableEmbeddings | None = None,
) -> tuple[Chroma, list[Document]]:
    docs = split_document(loaded)
    collection_name = collection_name_for(loaded.filename, loaded.text)
    persist_directory = Path(config.chroma_dir) / collection_name
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings or build_embeddings(config),
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )
    vector_store.persist()
    return vector_store, docs


def build_chain(
    config: AppConfig,
    vector_store: Chroma,
    *,
    documents: list[Document] | None = None,
    callback: Any | None = None,
) -> SimpleRagChain:
    return SimpleRagChain(
        config=config,
        vector_store=vector_store,
        documents=documents or [],
        callback=callback,
    )


def ask_document(
    chain: SimpleRagChain,
    question: str,
    chat_history: list[tuple[str, str]],
    *,
    callback: Any | None = None,
    config: AppConfig | None = None,
    session_id: str | None = None,
    document_name: str | None = None,
) -> RagAnswer:
    config = config or chain.config
    callback = callback or chain.callback

    with traced_observation(
        config,
        name="document-chat",
        as_type="chain",
        session_id=session_id,
        tags=["rag", "chat", "streamlit"],
        metadata={
            "document": document_name or "uploaded-document",
            "chat_model": config.chat_model,
            "embedding_model": config.embedding_model,
        },
        input={"question": question},
    ) as trace:
        with start_child_observation(
            config,
            name="retrieve-document-context",
            as_type="retriever",
            input={"question": question, "k": 4},
        ) as retrieval:
            source_documents = chain.vector_store.similarity_search(question, k=4)
            retrieval.update(output=_source_summary(source_documents))

        answer = answer_from_documents(
            question,
            chat_history,
            source_documents,
            config,
            callback=callback,
        )
        if trace is not None:
            trace.update(output={"answer": answer, "source_count": len(source_documents)})
        return RagAnswer(answer=answer, source_documents=source_documents)


def answer_from_documents(
    question: str,
    chat_history: list[tuple[str, str]],
    source_documents: list[Document],
    config: AppConfig,
    *,
    callback: Any | None = None,
) -> str:
    context = "\n\n".join(
        f"Source {index}:\n{doc.page_content}" for index, doc in enumerate(source_documents, start=1)
    )
    history = "\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in chat_history[-3:])
    prompt = f"""Answer the user's question using only the provided document context.
If the context does not contain the answer, say that the uploaded document does not provide enough information.

Conversation history:
{history or "None"}

Document context:
{context}

    Question: {question}
    Answer:"""
    response = build_llm(config).invoke(prompt, config=callback_config(callback))
    return extract_response_text(response)


def extract_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        if texts:
            return "\n".join(texts)
    return str(content)


def _source_summary(source_documents: list[Document]) -> dict[str, Any]:
    return {
        "source_count": len(source_documents),
        "sources": [
            {
                "source": doc.metadata.get("source", "uploaded document"),
                "type": doc.metadata.get("type", "document"),
                "characters": len(doc.page_content),
            }
            for doc in source_documents
        ],
    }


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ["500", "internal", "503", "unavailable", "deadline", "timeout"])
