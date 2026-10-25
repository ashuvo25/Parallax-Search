"""Answer questions from retrieved papers: retrieve -> build prompt -> generate."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import load_model  # noqa: E402
from src.llm import MODEL, generate  # noqa: E402
from src.llm import get_client as get_llm_client  # noqa: E402
from src.prompts import build_messages  # noqa: E402
from src.vector_store import get_client as get_chroma_client  # noqa: E402
from src.vector_store import get_collection, search  # noqa: E402

TOP_K = 5
MIN_SCORE = 0.3  # from the retrieval test: junk queries scored below 0.21


class RAGPipeline:
    """Loads the embedding model, ChromaDB and the LLM client once, then answers questions."""

    def __init__(self, k: int = TOP_K, min_score: float = MIN_SCORE):
        self.k = k
        self.min_score = min_score
        self.collection = get_collection(get_chroma_client())
        self.embedder = load_model()
        self.llm = get_llm_client()

    def answer(self, question: str) -> dict:
        results = search(self.collection, self.embedder, question, self.k, self.min_score)
        if not results:
            return {
                "answer": "I couldn't find any relevant papers for this question.",
                "sources": [],
                "llm": None,
            }

        resp = generate(build_messages(question, results), client=self.llm)
        sources = [
            {"n": i, "title": r["title"], "score": r["score"], "doc_id": r["doc_id"]}
            for i, r in enumerate(results, 1)
        ]
        return {"answer": resp.text, "sources": sources, "llm": resp}


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] ({s['score']:.3f}) {s['title']}")
    if result["llm"]:
        r = result["llm"]
        print(f"\nLLM: {r.model} | {r.latency_ms} ms | tokens in/out: "
              f"{r.prompt_tokens}/{r.completion_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about the arXiv papers.")
    parser.add_argument("question", nargs="?", help="leave empty for interactive mode")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    pipeline = RAGPipeline(k=args.k)
    print(f"Using {MODEL}")

    if args.question:
        print_result(pipeline.answer(args.question))
        return

    print("Ask a question (empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        print_result(pipeline.answer(question))


if __name__ == "__main__":
    main()