"""Answer questions from retrieved papers: retrieve -> build prompt -> generate."""
import argparse
import sys
import time
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
NO_RESULTS_ANSWER = "I couldn't find any relevant papers for this question."


class LLMNotConfigured(RuntimeError):
    """An answer needs the LLM, but no API client is available."""


class RAGPipeline:
    """Holds the embedding model, ChromaDB and the LLM client, loaded once.

    Any part can be passed in instead of loaded, which lets tests use fakes.
    """

    def __init__(self, k: int = TOP_K, min_score: float = MIN_SCORE,
                 collection=None, embedder=None, llm=None, load_llm: bool = True):
        self.k = k
        self.min_score = min_score
        self.collection = collection if collection is not None else get_collection(get_chroma_client())
        self.embedder = embedder if embedder is not None else load_model()
        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_llm_client() if load_llm else None

    def retrieve(self, question: str, k: int | None = None, where: dict | None = None) -> list[dict]:
        return search(self.collection, self.embedder, question, k or self.k, self.min_score, where)

    def answer(self, question: str, k: int | None = None, where: dict | None = None) -> dict:
        start = time.perf_counter()
        results = self.retrieve(question, k, where)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not results:
            return {
                "answer": NO_RESULTS_ANSWER,
                "sources": [],
                "llm": None,
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": 0.0,
                "total_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        if self.llm is None:
            raise LLMNotConfigured("LLM_API_KEY is not set, so answers can't be generated.")

        resp = generate(build_messages(question, results), client=self.llm)
        sources = [
            {"n": i, "title": r["title"], "score": r["score"], "doc_id": r["doc_id"]}
            for i, r in enumerate(results, 1)
        ]
        return {
            "answer": resp.text,
            "sources": sources,
            "llm": resp,
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": resp.latency_ms,
            "total_ms": round((time.perf_counter() - start) * 1000, 1),
        }


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] ({s['score']:.3f}) {s['title']}")
    print(f"\nTime: retrieval {result['retrieval_ms']} ms | generation "
          f"{result['generation_ms']} ms | total {result['total_ms']} ms")
    if result["llm"]:
        r = result["llm"]
        print(f"LLM: {r.model} | tokens in/out: {r.prompt_tokens}/{r.completion_tokens}")


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