"""
Chainlit chat UI — Phase 4.

Flow per message:
  1. User types a query
  2. POST /query  → receives { answer, trace_id, version, sources }
  3. Answer is displayed; thumbs up/down actions are registered on the message
  4. User clicks thumbs up or down → POST /feedback { trace_id, rating }
     → score attached to the Langfuse trace

The serving URL is read from the SERVING_URL environment variable
(default: http://localhost:8000 for local dev).
"""

from __future__ import annotations

import os

import httpx
import chainlit as cl

SERVING_URL = os.getenv("SERVING_URL", "http://localhost:8000").rstrip("/")
TOP_K = int(os.getenv("TOP_K", "5"))


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "Hi! I'm your Netflix assistant. Ask me anything about titles on Netflix "
            "— genres, plot summaries, recommendations, and more."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return

    # --- Call serving layer ---
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{SERVING_URL}/query",
                json={"query": query, "top_k": TOP_K},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            await cl.Message(content=f"Error reaching serving layer: {exc}").send()
            return

    data = resp.json()
    answer: str = data.get("answer", "(no answer returned)")
    trace_id: str = data.get("trace_id", "no-trace")
    version: str = data.get("version", "")

    # --- Display answer with feedback actions ---
    actions = [
        cl.Action(
            name="thumbs_up",
            icon="thumbs-up",
            label="",
            payload={"trace_id": trace_id, "rating": 1},
        ),
        cl.Action(
            name="thumbs_down",
            icon="thumbs-down",
            label="",
            payload={"trace_id": trace_id, "rating": 0},
        ),
    ]

    footer = f"\n\n*version: {version}*" if version else ""
    await cl.Message(content=answer + footer, actions=actions).send()


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    await _post_feedback(action.payload["trace_id"], action.payload["rating"])
    await action.remove()
    await cl.Message(content="Thanks for the feedback! 👍").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    await _post_feedback(action.payload["trace_id"], action.payload["rating"])
    await action.remove()
    await cl.Message(content="Thanks for the feedback! 👎").send()


async def _post_feedback(trace_id: str, rating: int) -> None:
    if trace_id == "no-trace":
        return
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(
                f"{SERVING_URL}/feedback",
                json={"trace_id": trace_id, "rating": rating},
            )
        except httpx.HTTPError:
            pass  # Feedback failure is non-critical
