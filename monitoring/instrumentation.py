"""
Langfuse tracing wrappers for the serving layer — SDK v3 compatible.

SDK v3 uses context-manager-based observations rather than the imperative
v2 trace/span object API. All methods are no-ops when Langfuse is disabled.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Tuple

logger = logging.getLogger("llmops.instrumentation")


class LangfuseClient:
    """
    Thin wrapper around the Langfuse SDK v3.

    Constructed once at API startup via from_config().
    Context managers yield (trace_id, obs) or just obs for spans.
    Disabled when keys are absent — all methods become no-ops.
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

    @contextmanager
    def trace_context(
        self, name: str, input: dict
    ) -> Generator[Tuple[str, Any], None, None]:
        """
        Context manager that wraps a full request as a root trace.

        Yields (trace_id, obs). When disabled, yields ("no-trace", None).
        Use get_current_trace_id() inside the block to retrieve the UUID.

        Example:
            with lf.trace_context("rag_query", {"query": q}) as (trace_id, obs):
                ...
                lf.update(obs, {"answer": answer})
        """
        if not self.enabled:
            yield "no-trace", None
            return
        try:
            with self._client.start_as_current_observation(
                as_type="span", name=name, input=input
            ) as obs:
                trace_id = self._client.get_current_trace_id() or obs.id
                yield trace_id, obs
        except Exception as exc:
            logger.warning("Langfuse trace_context failed: %s", exc)
            yield "no-trace", None

    @contextmanager
    def span_context(
        self, name: str, input: dict
    ) -> Generator[Any, None, None]:
        """
        Context manager for a child span.

        Must be called inside an active trace_context block so that v3's
        context variable links the span to the parent trace automatically.
        Yields the observation object (None when disabled).

        Example:
            with lf.span_context("retrieve", {"query": q}) as span:
                result = retriever.retrieve(q)
                lf.update(span, {"num_hits": len(result.items)})
        """
        if not self.enabled:
            yield None
            return
        try:
            with self._client.start_as_current_observation(
                as_type="span", name=name, input=input
            ) as obs:
                yield obs
        except Exception as exc:
            logger.warning("Langfuse span_context failed: %s", exc)
            yield None

    @contextmanager
    def generation_context(
        self, name: str, input: dict, model: str | None = None, metadata: dict | None = None
    ) -> Generator[Any, None, None]:
        """
        Context manager for an LLM generation observation (as_type='generation').

        Langfuse renders generation observations with prompt/completion diffs
        and tracks model/token usage separately from plain spans.
        Must be called inside an active trace_context block.

        Example:
            with lf.generation_context(
                "generate",
                input={"context": ctx, "question": q},
                metadata={"model": "gpt-4o-mini", "temperature": 0.2},
            ) as gen:
                answer = run_chain(...)
                lf.update(gen, {"answer": answer})
        """
        if not self.enabled:
            yield None
            return
        try:
            with self._client.start_as_current_observation(
                as_type="generation",
                name=name,
                input=input,
                model=model,
                metadata=metadata or {},
            ) as obs:
                yield obs
        except Exception as exc:
            logger.warning("Langfuse generation_context failed: %s", exc)
            yield None

    def update(
        self,
        obs: Any,
        output: dict | None = None,
        metadata: dict | None = None,
        usage: dict | None = None,
    ) -> None:
        """Set output, metadata, and/or usage on a trace or span observation."""
        if obs is None:
            return
        try:
            kwargs: dict = {}
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if usage is not None:
                kwargs["usage"] = usage
            obs.update(**kwargs)
        except Exception as exc:
            logger.warning("Langfuse update failed: %s", exc)

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)
