"""Configuration helpers for the Streamlit RAG app."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    google_api_key: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    chat_model: str
    embedding_model: str
    chroma_dir: str

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def load_config(
    *,
    google_api_key: str | None = None,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_host: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    chroma_dir: str | None = None,
) -> AppConfig:
    load_dotenv()

    return AppConfig(
        google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        langfuse_public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        chat_model=chat_model or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash"),
        embedding_model=embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        chroma_dir=chroma_dir or os.getenv("CHROMA_DIR", "chroma_db"),
    )
