"""Langfuse tracing helpers."""

from __future__ import annotations

import os
import base64
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
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "local-streamlit")


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
        span_exporter=_build_langfuse_span_exporter(config),
    )
    return get_client(public_key=config.langfuse_public_key)


def build_langfuse_callback(config: AppConfig) -> Any | None:
    if not config.langfuse_enabled:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError("langfuse is not installed. Run pip install -r requirements.txt.") from exc

    configure_langfuse_environment(config)
    get_langfuse_client(config)
    return CallbackHandler(public_key=config.langfuse_public_key)


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


def _build_langfuse_span_exporter(config: AppConfig) -> Any | None:
    if not config.langfuse_enabled:
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    headers = {
        "Authorization": "Basic "
        + base64.b64encode(f"{config.langfuse_public_key}:{config.langfuse_secret_key}".encode("utf-8")).decode(
            "ascii"
        ),
        "x-langfuse-sdk-name": "python",
        "x-langfuse-public-key": config.langfuse_public_key or "",
    }
    exporter = OTLPSpanExporter(
        endpoint=f"{config.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
        headers=headers,
        timeout=10,
    )

    if _verify_ssl_enabled(config):
        try:
            import certifi

            exporter._certificate_file = certifi.where()
        except ImportError:
            pass
    else:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        exporter._certificate_file = False

    return exporter


def _verify_ssl_enabled(config: AppConfig) -> bool:
    return config.langfuse_verify_ssl


class _NoopObservation:
    def __enter__(self) -> "_NoopObservation":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def update(self, **_: Any) -> None:
        return None
