"""
Posts user feedback scores to Langfuse.

Called by POST /feedback after the client receives a trace_id from /query.
"""

from __future__ import annotations

import logging

from monitoring.instrumentation import LangfuseClient

logger = logging.getLogger("llmops.feedback")


def post_feedback(client: LangfuseClient, trace_id: str, rating: int) -> None:
    """
    Attach a user rating score to an existing Langfuse trace.

    rating: 1 = thumbs up, 0 = thumbs down.
    Silently skips if Langfuse is disabled or the trace_id is a placeholder.
    """
    if not client.enabled or trace_id == "no-trace":
        return
    try:
        client._client.score(
            trace_id=trace_id,
            name="user_rating",
            value=rating,
        )
    except Exception as exc:
        logger.warning("Langfuse post_feedback failed (trace_id=%s): %s", trace_id, exc)
