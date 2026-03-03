"""
RAG generation chain.

Retrieval results are formatted into a context block, then passed together
with the user's query to gpt-4o-mini via LangChain's ChatOpenAI.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_SYSTEM_PROMPT = """\
You are a helpful Netflix content assistant.

Use ONLY the retrieved titles below to answer the user's question.
Do not invent titles or details that are not in the context.
If the context does not contain enough information, say so clearly and \
suggest what kind of titles the user might search for instead.

Retrieved titles:
{context}"""


def _format_context(items: List[Dict]) -> str:
    lines: List[str] = []
    for item in items:
        title = item.get("title", "Unknown")
        kind = item.get("type", "")
        desc = item.get("description", "")
        line = f"- {title}"
        if kind:
            line += f" ({kind})"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no results retrieved)"


def run_chain(
    query: str,
    items: List[Dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Run the RAG generation step.

    items: retrieval results from Retriever.retrieve().items
    Returns the model's answer as a plain string.
    """
    context = _format_context(items)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=model, temperature=temperature)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})
