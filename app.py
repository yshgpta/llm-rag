from __future__ import annotations

import uuid

import streamlit as st

from src.config import load_config
from src.document_loader import load_uploaded_document
from src.evaluation import EvaluationRow, run_ragas_evaluation, source_contexts
from src.monitoring import build_langfuse_callback, check_langfuse_connection, flush_langfuse, traced_observation
from src.rag import ask_document, build_chain, create_vector_store


st.set_page_config(page_title="Gemini Document RAG", page_icon="📄", layout="wide")


def init_state() -> None:
    defaults = {
        "session_id": str(uuid.uuid4()),
        "loaded_document": None,
        "vector_store": None,
        "chain": None,
        "documents": [],
        "chat_history": [],
        "messages": [],
        "eval_rows": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar():
    env_config = load_config()
    with st.sidebar:
        st.header("Settings")
        google_api_key = st.text_input(
            "Gemini API key",
            value=env_config.google_api_key,
            type="password",
            help="Used for Gemini chat and embeddings.",
        )
        langfuse_public_key = st.text_input(
            "Langfuse public key",
            value=env_config.langfuse_public_key or "",
            type="password",
        )
        langfuse_secret_key = st.text_input(
            "Langfuse secret key",
            value=env_config.langfuse_secret_key or "",
            type="password",
        )
        langfuse_host = st.text_input("Langfuse host", value=env_config.langfuse_host)
        langfuse_environment = st.selectbox(
            "Langfuse environment",
            options=["streamlit", "local-streamlit", "default"],
            index=_option_index(["streamlit", "local-streamlit", "default"], env_config.langfuse_environment),
            help="Use the same environment filter in the Langfuse dashboard.",
        )
        chat_model = st.selectbox(
            "Gemini chat model",
            options=[
                "gemini-3.5-flash",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-flash-latest",
            ],
            index=_option_index(
                ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"],
                env_config.chat_model,
            ),
            key="chat_model_v2",
        )
        embedding_model = st.selectbox(
            "Gemini embedding model",
            options=[
                "models/gemini-embedding-001",
                "models/gemini-embedding-2",
            ],
            index=_option_index(
                ["models/gemini-embedding-001", "models/gemini-embedding-2"],
                env_config.embedding_model,
            ),
            key="embedding_model_v2",
        )
        chroma_dir = st.text_input("Chroma directory", value=env_config.chroma_dir)

        st.divider()
        st.caption("Use `.env` locally or Streamlit secrets when deployed.")

    config = load_config(
        google_api_key=google_api_key or None,
        langfuse_public_key=langfuse_public_key or None,
        langfuse_secret_key=langfuse_secret_key or None,
        langfuse_host=langfuse_host or None,
        langfuse_environment=langfuse_environment or None,
        chat_model=chat_model or None,
        embedding_model=embedding_model or None,
        chroma_dir=chroma_dir or None,
    )
    with st.sidebar:
        if config.langfuse_enabled:
            st.success(f"Langfuse configured for `{config.langfuse_host}` / `{config.langfuse_environment}`.")
            if st.button("Test Langfuse connection", width="stretch"):
                ok, message = check_langfuse_connection(config)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.info("Langfuse tracing is off until both Langfuse keys are set.")

    return config


def _option_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def render_upload(config) -> None:
    uploaded_file = st.file_uploader("Upload a PDF, JPG, JPEG, or PNG", type=["pdf", "jpg", "jpeg", "png"])
    can_index = uploaded_file is not None and bool(config.google_api_key)

    cols = st.columns([1, 2])
    with cols[0]:
        index_clicked = st.button("Index document", disabled=not can_index, width="stretch")
    with cols[1]:
        if uploaded_file and not config.google_api_key:
            st.warning("Add a Gemini API key before indexing.")

    if not index_clicked:
        return

    callback = build_langfuse_callback(config)

    with st.status("Reading and indexing document...", expanded=True) as status:
        try:
            with traced_observation(
                config,
                name="document-index",
                as_type="chain",
                session_id=st.session_state.session_id,
                tags=["rag", "index", "streamlit"],
                metadata={"filename": uploaded_file.name},
                input={"filename": uploaded_file.name, "content_type": uploaded_file.type},
            ) as trace:
                loaded = load_uploaded_document(uploaded_file, uploaded_file.name)
                st.write(f"Loaded `{loaded.filename}` with {loaded.page_count} page(s).")
                vector_store, docs = create_vector_store(loaded, config)
                st.write(f"Created {len(docs)} chunks in Chroma.")
                chain = build_chain(config, vector_store, documents=docs, callback=callback)
                if trace is not None:
                    trace.update(
                        output={
                            "filename": loaded.filename,
                            "page_count": loaded.page_count,
                            "chunk_count": len(docs),
                        }
                    )
                status.update(label="Document ready for chat.", state="complete")
        finally:
            flush_langfuse(config)

    st.session_state.loaded_document = loaded
    st.session_state.vector_store = vector_store
    st.session_state.chain = chain
    st.session_state.documents = docs
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.eval_rows = []
    st.success("Indexing complete. Ask questions below.")


def render_chat(config) -> None:
    chain = st.session_state.chain
    loaded = st.session_state.loaded_document

    st.subheader("Chat")
    if not chain:
        st.info("Upload and index a document to start chatting.")
        return

    st.caption(f"Current document: `{loaded.filename}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about the uploaded document")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    callback = build_langfuse_callback(config)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                response = ask_document(
                    chain,
                    question,
                    st.session_state.chat_history,
                    callback=callback,
                    config=config,
                    session_id=st.session_state.session_id,
                    document_name=loaded.filename,
                )
            except Exception as exc:
                flush_langfuse(config)
                message = str(exc)
                st.error(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
                return
            flush_langfuse(config)
            st.markdown(response.answer)

            with st.expander("Sources"):
                for index, doc in enumerate(response.source_documents, start=1):
                    source = doc.metadata.get("source", "uploaded document")
                    st.markdown(f"**Source {index}: {source}**")
                    st.write(doc.page_content[:1200])

    st.session_state.messages.append({"role": "assistant", "content": response.answer})
    st.session_state.chat_history.append((question, response.answer))
    st.session_state.eval_rows.append(
        EvaluationRow(
            question=question,
            answer=response.answer,
            contexts=source_contexts(response.source_documents),
        )
    )


def render_evaluation(config) -> None:
    st.subheader("Evaluation")
    if not st.session_state.eval_rows:
        st.caption("Ask at least one question to create evaluation rows.")
        return

    st.write(f"{len(st.session_state.eval_rows)} chat turn(s) ready for RAGAS evaluation.")
    if st.button("Run RAGAS evaluation", width="content"):
        with st.spinner("Running RAGAS metrics..."):
            try:
                result = run_ragas_evaluation(st.session_state.eval_rows, config)
            except Exception as exc:
                st.error(f"RAGAS evaluation failed: {exc}")
                return
        st.dataframe(result, width="stretch")


def main() -> None:
    init_state()
    config = render_sidebar()

    st.title("Gemini Document RAG")
    st.write("Upload a document, create a Chroma-backed RAG index, and chat with Gemini using document-grounded answers.")

    upload_tab, chat_tab, eval_tab = st.tabs(["Document", "Chat", "Evaluate"])
    with upload_tab:
        render_upload(config)
    with chat_tab:
        render_chat(config)
    with eval_tab:
        render_evaluation(config)


if __name__ == "__main__":
    main()
