"""Prompt building for the RAG pipeline."""

SYSTEM_PROMPT = (
    "You answer questions about AI research papers using only the context provided. "
    "If the context does not contain the answer, say that you could not find it. "
    "Cite the papers you use by their number in square brackets, like [1]."
)


def _body(result: dict) -> str:
    """Document text without the title at the start (the title is shown separately)."""
    title, text = result.get("title", ""), result.get("text", "")
    return text[len(title):].strip() if title and text.startswith(title) else text


def format_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {r.get('title', 'Untitled')}\n{_body(r)}"
        for i, r in enumerate(results, 1)
    )


def build_messages(question: str, results: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{format_context(results)}\n\nQuestion: {question}"},
    ]