from src.document_loader import LoadedDocument
from src.evaluation import EvaluationRow, _build_ragas_metrics, build_evaluation_dataset, source_contexts
from src.config import load_config
from src.rag import answer_from_documents, collection_name_for, extract_response_text, split_document
from langchain_core.documents import Document


def test_split_document_preserves_source_metadata():
    loaded = LoadedDocument(
        filename="policy.pdf",
        text="Acko health policy covers hospitalization. " * 80,
        page_count=1,
        source_type="pdf",
    )

    docs = split_document(loaded, chunk_size=120, chunk_overlap=20)

    assert len(docs) > 1
    assert all(doc.metadata["source"] == "policy.pdf" for doc in docs)
    assert all(doc.metadata["type"] == "pdf" for doc in docs)


def test_collection_name_is_stable_and_chroma_safe():
    first = collection_name_for("policy.pdf", "hello")
    second = collection_name_for("policy.pdf", "hello")

    assert first == second
    assert first.startswith("doc_")
    assert 3 <= len(first) <= 63


def test_source_contexts_limits_documents():
    docs = [Document(page_content=f"context {index}") for index in range(4)]

    contexts = source_contexts(docs, limit=2)

    assert len(contexts) == 2


def test_source_contexts_trims_long_contexts():
    docs = [Document(page_content="word " * 500)]

    contexts = source_contexts(docs)

    assert len(contexts[0]) <= 1600


def test_build_evaluation_dataset_shape():
    dataset = build_evaluation_dataset(
        [
            EvaluationRow(
                question="What is covered?",
                answer="Hospitalization is covered.",
                contexts=["Acko health policy covers hospitalization."],
            )
        ]
    )

    assert dataset.num_rows == 1
    assert set(dataset.column_names) == {"user_input", "response", "retrieved_contexts"}


def test_ragas_metrics_use_async_gemini_llm_and_google_embeddings():
    config = load_config(google_api_key="fake", chat_model="gemini-test", embedding_model="models/embed-test")
    calls = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            calls.append(("async_openai", kwargs))

    class FakeGoogleGenAIClient:
        def __init__(self, **kwargs):
            calls.append(("google_genai", kwargs))

    class FakeGoogleEmbeddings:
        def __init__(self, **kwargs):
            calls.append(("google_embeddings", kwargs))

    def fake_llm_factory(model, **kwargs):
        calls.append(("llm_factory", model, kwargs))
        return object()

    class FakeFaithfulness:
        def __init__(self, **kwargs):
            calls.append(("faithfulness", sorted(kwargs)))

    class FakeAnswerRelevancy:
        def __init__(self, **kwargs):
            calls.append(("answer_relevancy", kwargs["strictness"]))

    metrics = _build_ragas_metrics(
        config,
        AsyncOpenAI=FakeAsyncOpenAI,
        GoogleGenAIClient=FakeGoogleGenAIClient,
        GoogleEmbeddings=FakeGoogleEmbeddings,
        llm_factory=fake_llm_factory,
        Faithfulness=FakeFaithfulness,
        AnswerRelevancy=FakeAnswerRelevancy,
    )

    assert set(metrics) == {"faithfulness", "answer_relevancy"}
    assert (
        "async_openai",
        {"api_key": "fake", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    ) in calls
    assert ("google_genai", {"api_key": "fake"}) in calls
    assert any(
        call[0] == "llm_factory" and call[1] == "gemini-test" and call[2]["max_tokens"] == 8192
        for call in calls
    )
    assert any(call[0] == "google_embeddings" and call[1]["model"] == "embed-test" for call in calls)


def test_answer_prompt_includes_recent_chat_history(monkeypatch):
    docs = [Document(page_content="Experience: Yash has 5 years as a software engineer.")]
    config = load_config(google_api_key="fake")
    captured = {}

    class FakeResponse:
        content = "Yash has 5 years of experience."

    class FakeLlm:
        def invoke(self, prompt, config=None):
            captured["prompt"] = prompt
            return FakeResponse()

    monkeypatch.setattr("src.rag.build_llm", lambda config: FakeLlm())

    answer = answer_from_documents("What about his experience?", [("Who is this?", "This is Yash.")], docs, config)

    assert "5 years" in answer
    assert "User: Who is this?" in captured["prompt"]
    assert "Question: What about his experience?" in captured["prompt"]


def test_extract_response_text_ignores_signature_payload():
    class Response:
        content = [
            {
                "type": "text",
                "text": "The uploaded document does not provide enough information.",
                "extras": {"signature": "opaque-signature"},
            }
        ]

    assert extract_response_text(Response()) == "The uploaded document does not provide enough information."
