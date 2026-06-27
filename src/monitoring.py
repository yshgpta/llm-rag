"""Langfuse tracing helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .config import AppConfig


def configure_langfuse_environment(config: AppConfig) -> None:
    """Set SDK environment variables before the Langfuse client is initialized."""
    if not config.langfuse_enabled:
        return

    try:
        import certifi

        cert_path = certifi.where()
        os.environ["SSL_CERT_FILE"] = cert_path
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path
        os.environ["CURL_CA_BUNDLE"] = cert_path
    except ImportError:
        pass

    os.environ["LANGFUSE_PUBLIC_KEY"] = config.langfuse_public_key or ""
    os.environ["LANGFUSE_SECRET_KEY"] = config.langfuse_secret_key or ""
    os.environ["LANGFUSE_HOST"] = config.langfuse_host
    os.environ["LANGFUSE_BASE_URL"] = config.langfuse_host
    os.environ["LANGFUSE_VERIFY_SSL"] = "true" if config.langfuse_verify_ssl else "false"
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "streamlit")


def get_langfuse_client(config: AppConfig) -> Any | None:
    if not config.langfuse_enabled:
        return None

    try:
        from langfuse import Langfuse, get_client
    except ImportError as exc:
        raise RuntimeError("langfuse is not installed. Run pip install -r requirements.txt.") from exc

    configure_langfuse_environment(config)
    Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host,
        httpx_client=_build_langfuse_httpx_client(config),
        environment=os.environ["LANGFUSE_TRACING_ENVIRONMENT"],
    )
    return get_client()


def build_langfuse_callback(config: AppConfig) -> Any | None:
    if not config.langfuse_enabled:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError("langfuse is not installed. Run pip install -r requirements.txt.") from exc

    configure_langfuse_environment(config)
    get_langfuse_client(config)
    return CallbackHandler()


def callback_config(callback: Any | None) -> dict[str, list[Any]]:
    return {"callbacks": [callback]} if callback else {}


@contextmanager
def traced_observation(
    config: AppConfig,
    *,
    name: str,
    as_type: str = "span",
    session_id: str | None = None,
    user_id: str | None = "streamlit-user",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any | None = None,
) -> Iterator[Any | None]:
    """Create a Langfuse root/current observation and propagate trace attributes."""
    client = get_langfuse_client(config)
    if client is None:
        yield None
        return

    try:
        from langfuse import propagate_attributes
    except ImportError as exc:
        raise RuntimeError("langfuse is not installed. Run pip install -r requirements.txt.") from exc

    with client.start_as_current_observation(as_type=as_type, name=name, input=input) as observation:
        with propagate_attributes(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            metadata=_string_metadata(metadata or {}),
        ):
            yield observation


def start_child_observation(config: AppConfig, *, name: str, as_type: str = "span", **kwargs: Any) -> Any:
    client = get_langfuse_client(config)
    if client is None:
        return _NoopObservation()
    return client.start_as_current_observation(as_type=as_type, name=name, **kwargs)


def flush_langfuse(config: AppConfig) -> None:
    client = get_langfuse_client(config)
    if client is not None:
        client.flush()


def check_langfuse_connection(config: AppConfig) -> tuple[bool, str]:
    if not config.langfuse_enabled:
        return False, "Langfuse public and secret keys are not configured."

    try:
        client = get_langfuse_client(config)
        if client is None:
            return False, "Langfuse client is disabled."
        if client.auth_check():
            return True, f"Langfuse authentication succeeded for {config.langfuse_host}."
        return False, f"Langfuse authentication failed for {config.langfuse_host}."
    except Exception as exc:
        return False, f"Langfuse connection failed: {exc}"


def _string_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in metadata.items():
        normalized_key = "".join(character for character in str(key) if character.isalnum() or character == "_")
        if not normalized_key:
            continue
        clean[normalized_key] = str(value)[:200]
    return clean


def _build_langfuse_httpx_client(config: AppConfig) -> Any | None:
    if _verify_ssl_enabled(config):
        return None

    import httpx

    return httpx.Client(verify=False)


def _verify_ssl_enabled(config: AppConfig) -> bool:
    return config.langfuse_verify_ssl


class _NoopObservation:
    def __enter__(self) -> "_NoopObservation":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, **_: Any) -> None:
        return None
