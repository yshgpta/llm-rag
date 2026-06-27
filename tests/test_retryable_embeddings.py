import pytest

from src.rag import RetryableEmbeddings


class FlakyEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("500 INTERNAL")
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class BrokenEmbeddings:
    def embed_query(self, text):
        raise RuntimeError("500 INTERNAL")

    def embed_documents(self, texts):
        raise RuntimeError("500 INTERNAL")


def test_retryable_embeddings_retries_transient_query_failure():
    embeddings = FlakyEmbeddings()
    retrying = RetryableEmbeddings(embeddings, attempts=2, delay_seconds=0)

    assert retrying.embed_query("hello") == [1.0, 0.0]
    assert embeddings.calls == 2


def test_retryable_embeddings_raises_friendly_error_after_retries():
    retrying = RetryableEmbeddings(BrokenEmbeddings(), attempts=2, delay_seconds=0)

    with pytest.raises(RuntimeError, match="Gemini embedding service failed"):
        retrying.embed_query("hello")
