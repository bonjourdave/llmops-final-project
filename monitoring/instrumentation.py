"""
Langfuse tracing wrappers for the serving layer.

Every /query request gets one trace. The retrieval step gets a child span
inside that trace. Telemetry failures are caught and logged — they never
crash the serving layer.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger("llmops.instrumentation")


class LangfuseClient:
    """
    Thin wrapper around the Langfuse SDK.

    Constructed once at API startup via from_config().
    All methods are no-ops when Langfuse is disabled or unavailable.
    """

    def __init__(self, client: Any | None) -> None:
        self._client = client

    @classmethod
    def from_config(cls, cfg: dict) -> "LangfuseClient":
        """
        Build a LangfuseClient from a loaded monitoring.yaml dict.

        cfg expected shape:
            langfuse:
              host: "..."
              public_key: "..."
              secret_key: "..."
        """
        lf_cfg = cfg.get("langfuse", {})
        public_key = lf_cfg.get("public_key", "")
        secret_key = lf_cfg.get("secret_key", "")
        host = lf_cfg.get("host", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.info("Langfuse disabled: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set.")
            return cls(None)

        try:
            from langfuse import Langfuse

            client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            logger.info("Langfuse enabled (host=%s)", host)
            return cls(client)
        except Exception as exc:
            logger.warning("Langfuse init failed — tracing disabled: %s", exc)
            return cls(None)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_trace(self, name: str, input: dict) -> Tuple[str, Any]:
        """
        Open a new trace and return (trace_id, trace_handle).
        Returns ("no-trace", None) when disabled.
        """
        if not self.enabled:
            return "no-trace", None
        try:
            trace = self._client.trace(name=name, input=input)
            return trace.id, trace
        except Exception as exc:
            logger.warning("Langfuse start_trace failed: %s", exc)
            return "no-trace", None

    def span(self, trace: Any, name: str, input: dict) -> Any:
        """Open a child span on an existing trace handle."""
        if trace is None:
            return None
        try:
            return trace.span(name=name, input=input)
        except Exception as exc:
            logger.warning("Langfuse span failed: %s", exc)
            return None

    def end_span(self, span: Any, output: dict) -> None:
        if span is None:
            return
        try:
            span.end(output=output)
        except Exception as exc:
            logger.warning("Langfuse end_span failed: %s", exc)

    def end_trace(self, trace: Any, output: dict) -> None:
        if trace is None:
            return
        try:
            trace.update(output=output)
        except Exception as exc:
            logger.warning("Langfuse end_trace failed: %s", exc)

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)
