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
    langfuse_verify_ssl: bool
    langfuse_environment: str
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
    langfuse_verify_ssl: bool | str | None = None,
    langfuse_environment: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    chroma_dir: str | None = None,
) -> AppConfig:
    load_dotenv()

    return AppConfig(
        google_api_key=google_api_key or _setting("GOOGLE_API_KEY", ""),
        langfuse_public_key=langfuse_public_key or _setting("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=langfuse_secret_key or _setting("LANGFUSE_SECRET_KEY"),
        langfuse_host=langfuse_host or _setting("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        langfuse_verify_ssl=_as_bool(
            langfuse_verify_ssl if langfuse_verify_ssl is not None else _setting("LANGFUSE_VERIFY_SSL", "true")
        ),
        langfuse_environment=langfuse_environment or _setting("LANGFUSE_TRACING_ENVIRONMENT", "streamlit"),
        chat_model=chat_model or _setting("GEMINI_CHAT_MODEL", "gemini-3.5-flash"),
        embedding_model=embedding_model or _setting("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        chroma_dir=chroma_dir or _setting("CHROMA_DIR", "chroma_db"),
    )


def _setting(name: str, default: str | None = None) -> str | None:
    """Read config from environment first, then Streamlit secrets."""
    value = os.getenv(name)
    if value not in {None, ""}:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value in {None, ""}:
        return default
    return str(secret_value)


def _as_bool(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "true").lower() not in {"0", "false", "no"}
