"""Optional RAGAS evaluation helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
import sys
import types

from langchain_core.documents import Document

from .config import AppConfig


@dataclass(frozen=True)
class EvaluationRow:
    question: str
    answer: str
    contexts: list[str]


def build_evaluation_dataset(rows: list[EvaluationRow]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required by ragas. Install requirements.txt first.") from exc

    return Dataset.from_dict(
        {
            "user_input": [row.question for row in rows],
            "response": [row.answer for row in rows],
            "retrieved_contexts": [row.contexts for row in rows],
        }
    )


def source_contexts(source_documents: list[Document], *, limit: int = 4) -> list[str]:
    return [_truncate_text(doc.page_content, max_characters=1600) for doc in source_documents[:limit]]


def run_local_evaluation(rows: list[EvaluationRow]) -> list[dict[str, object]]:
    results = []
    for index, row in enumerate(rows, start=1):
        question_tokens = _content_tokens(row.question)
        answer_tokens = _content_tokens(row.answer)
        context_tokens = _content_tokens(" ".join(row.contexts))
        answer_supported = _overlap(answer_tokens, context_tokens)
        question_answer_overlap = _overlap(question_tokens, answer_tokens)
        results.append(
            {
                "turn": index,
                "question": row.question,
                "contexts": len(row.contexts),
                "context_coverage": round(answer_supported, 4),
                "question_answer_overlap": round(question_answer_overlap, 4),
            }
        )
    return results


def run_ragas_evaluation(rows: list[EvaluationRow], config: AppConfig) -> list[dict[str, object]]:
    if not config.google_api_key:
        raise ValueError("Google API key is required to run RAGAS evaluation.")

    _install_ragas_vertexai_compat()
    try:
        from google import genai
        from openai import AsyncOpenAI
        from ragas.embeddings.google_provider import GoogleEmbeddings
        from ragas.llms.base import llm_factory
        from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
        from ragas.metrics.collections.faithfulness import Faithfulness
    except ImportError as exc:
        raise RuntimeError(f"RAGAS could not be imported: {exc}") from exc

    metrics = _build_ragas_metrics(
        config,
        AsyncOpenAI=AsyncOpenAI,
        GoogleGenAIClient=genai.Client,
        GoogleEmbeddings=GoogleEmbeddings,
        llm_factory=llm_factory,
        Faithfulness=Faithfulness,
        AnswerRelevancy=AnswerRelevancy,
    )
    return _run_async_evaluation(rows, metrics)


def _build_ragas_metrics(config: AppConfig, **deps: object) -> dict[str, object]:
    llm_client = deps["AsyncOpenAI"](
        api_key=config.google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    embedding_client = deps["GoogleGenAIClient"](api_key=config.google_api_key)
    llm = deps["llm_factory"](
        config.chat_model,
        client=llm_client,
        temperature=0,
        max_tokens=8192,
    )
    embeddings = deps["GoogleEmbeddings"](
        client=embedding_client,
        model=config.embedding_model.removeprefix("models/"),
    )
    return {
        "faithfulness": deps["Faithfulness"](llm=llm),
        "answer_relevancy": deps["AnswerRelevancy"](llm=llm, embeddings=embeddings, strictness=1),
    }


def _run_async_evaluation(rows: list[EvaluationRow], metrics: dict[str, object]) -> list[dict[str, object]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_score_rows(rows, metrics))

    thread_result: list[dict[str, object]] | None = None
    thread_error: BaseException | None = None

    def run_in_thread() -> None:
        nonlocal thread_result, thread_error
        try:
            thread_result = asyncio.run(_score_rows(rows, metrics))
        except BaseException as exc:
            thread_error = exc

    import threading

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    if thread_error:
        raise thread_error
    return thread_result or []


async def _score_rows(rows: list[EvaluationRow], metrics: dict[str, object]) -> list[dict[str, object]]:
    scores = []
    for index, row in enumerate(rows, start=1):
        score_row: dict[str, object] = {
            "turn": index,
            "question": row.question,
            "contexts": len(row.contexts),
        }
        faithfulness = await metrics["faithfulness"].ascore(
            user_input=row.question,
            response=_truncate_text(row.answer, max_characters=2400),
            retrieved_contexts=[_truncate_text(context, max_characters=1600) for context in row.contexts[:4]],
        )
        answer_relevancy = await metrics["answer_relevancy"].ascore(
            user_input=row.question,
            response=_truncate_text(row.answer, max_characters=2400),
        )
        score_row["faithfulness"] = round(float(faithfulness.value), 4)
        score_row["answer_relevancy"] = round(float(answer_relevancy.value), 4)
        scores.append(score_row)
    return scores


def _install_ragas_vertexai_compat() -> None:
    legacy_module = "langchain_community.chat_models.vertexai"
    if legacy_module in sys.modules:
        return

    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        return

    shim = types.ModuleType(legacy_module)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[legacy_module] = shim


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stopwords}


def _truncate_text(text: str, *, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    return text[:max_characters].rsplit(" ", 1)[0]


def _overlap(source: set[str], target: set[str]) -> float:
    if not source:
        return 0.0
    return len(source & target) / len(source)
