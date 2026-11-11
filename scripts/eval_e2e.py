"""End-to-end evaluation of the RAG system: answer quality and latency.

Every question in eval/e2e_questions.csv goes through the full pipeline (in-process,
or through the running API with --api), and these are measured:
  behavior      answers in-domain questions, refuses off-topic and out-of-corpus ones
  citations     cites sources, and only sources that exist
  groundedness  share of answer sentences supported by the retrieved papers
  relevance     similarity between question and answer
  judge         optional LLM-as-judge scores for faithfulness and relevance (--judge)
  latency       retrieval, generation and total time, plus tokens
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import load_model  # noqa: E402
from src.llm import LLMError, generate  # noqa: E402
from src.llm import get_client as get_llm_client  # noqa: E402
from src.prompts import format_context  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402
from src.vector_store import CORPUS_FILE  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "e2e_questions.csv"
LOG_DIR = ROOT / "logs"
SUMMARY_FILE = LOG_DIR / "e2e_eval.json"
RESULTS_FILE = LOG_DIR / "e2e_eval_results.csv"
ANSWERS_FILE = LOG_DIR / "e2e_eval_answers.jsonl"

SUPPORT_THRESHOLD = 0.5      # min similarity for an answer sentence to count as supported
EST_TOKENS_PER_CALL = 2500   # used to stay under the provider's tokens-per-minute limit
REFUSAL_PHRASES = (
    "couldn't find", "could not find", "no relevant papers", "does not contain",
    "doesn't contain", "do not contain", "not mentioned in the", "cannot answer",
    "can't answer", "not covered", "no information", "not provided in the",
    "none of the provided", "not in the provided",
)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# ---------- rate limiting ----------

class TokenBudget:
    """Waits when the next call would go over the provider's tokens-per-minute limit."""

    def __init__(self, tpm: int):
        self.tpm = tpm
        self.used = deque()  # (time, tokens)

    def wait(self, estimate: int) -> None:
        if not self.tpm:
            return
        while True:
            now = time.monotonic()
            while self.used and now - self.used[0][0] > 60:
                self.used.popleft()
            if not self.used or sum(t for _, t in self.used) + estimate <= self.tpm:
                return
            pause = max(1.0, 60 - (now - self.used[0][0]))
            print(f"  waiting {pause:.0f}s for the token limit...")
            time.sleep(pause)

    def record(self, tokens: int | None) -> None:
        self.used.append((time.monotonic(), tokens or 0))


# ---------- ways to ask ----------

class InProcessRunner:
    """Calls the RAG pipeline directly."""

    def __init__(self):
        self.pipeline = RAGPipeline()
        self.embedder = self.pipeline.embedder
        self.llm = self.pipeline.llm

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        try:
            r = self.pipeline.answer(question)
        except LLMError as e:
            return {"error": type(e).__name__, "message": str(e),
                    "client_ms": round((time.perf_counter() - start) * 1000, 1)}
        llm = r["llm"]
        return {
            "answer": r["answer"],
            "sources": r["sources"],
            "llm_called": llm is not None,
            "retrieval_ms": r["retrieval_ms"],
            "generation_ms": r["generation_ms"],
            "total_ms": r["total_ms"],
            "client_ms": round((time.perf_counter() - start) * 1000, 1),
            "prompt_tokens": llm.prompt_tokens if llm else 0,
            "completion_tokens": llm.completion_tokens if llm else 0,
        }


class ApiRunner:
    """Sends real HTTP requests to the running FastAPI server."""

    def __init__(self, base_url: str):
        import httpx
        self.http = httpx.Client(base_url=base_url, timeout=120)
        try:
            self.http.get("/health").raise_for_status()
        except httpx.HTTPError:
            sys.exit(f"API not reachable at {base_url}. Start it with: "
                     f"uvicorn src.api.main:app --no-access-log")
        self.embedder = load_model()
        self.llm = None

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        resp = self.http.post("/ask", json={"question": question})
        client_ms = round((time.perf_counter() - start) * 1000, 1)

        if resp.status_code != 200:
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            return {"error": err.get("type", f"http_{resp.status_code}"),
                    "message": err.get("message", resp.text[:200]), "client_ms": client_ms}

        d = resp.json()
        tokens = d.get("tokens") or {}
        return {
            "answer": d["answer"],
            "sources": d["sources"],
            "llm_called": d["llm_called"],
            "retrieval_ms": d["latency"]["retrieval_ms"],
            "generation_ms": d["latency"]["generation_ms"],
            "total_ms": d["latency"]["total_ms"],
            "client_ms": client_ms,
            "prompt_tokens": tokens.get("prompt") or 0,
            "completion_tokens": tokens.get("completion") or 0,
        }


# ---------- quality checks ----------

def is_refusal(answer: str, llm_called: bool) -> bool:
    if not llm_called:
        return True  # nothing retrieved, so the LLM was never asked
    text = answer.lower()
    return any(p in text for p in REFUSAL_PHRASES)


def find_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = []
    for group in CITATION.findall(answer):
        cited += [int(x) for x in re.split(r"\s*,\s*", group)]
    invalid = [c for c in cited if c < 1 or c > n_sources]
    return cited, invalid


def split_sentences(text: str, min_words: int) -> list[str]:
    text = CITATION.sub("", text)
    text = re.sub(r"[*#`|>]", " ", text)  # markdown leftovers
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(p.split()) for p in parts if len(p.split()) >= min_words]


def groundedness(embedder, answer: str, source_texts: list[str],
                 threshold: float) -> tuple[float | None, float | None, list[str]]:
    """Share of answer sentences whose best match in the sources reaches the threshold."""
    answer_sents = split_sentences(answer, min_words=6)
    source_sents = [s for t in source_texts for s in split_sentences(t, min_words=4)]
    if not answer_sents or not source_sents:
        return None, None, []

    a = embedder.encode(answer_sents, normalize_embeddings=True)
    s = embedder.encode(source_sents, normalize_embeddings=True)
    best = (a @ s.T).max(axis=1)
    unsupported = [sent for sent, score in zip(answer_sents, best) if score < threshold]
    return float((best >= threshold).mean()), float(best.mean()), unsupported


def answer_relevance(embedder, question: str, answer: str) -> float:
    q, a = embedder.encode([question, answer], normalize_embeddings=True)
    return float(q @ a)


JUDGE_SYSTEM = ("You grade answers produced by a retrieval-augmented system. "
                "Reply with one JSON object and nothing else.")


def llm_judge(client, question: str, answer: str, sources: list[dict]) -> tuple[dict, int]:
    prompt = (
        f"Sources:\n{format_context(sources)}\n\n"
        f"Question: {question}\n\nAnswer:\n{answer}\n\n"
        "Score the answer:\n"
        "- faithfulness 1-5: 5 means every claim is supported by the sources\n"
        "- relevance 1-5: 5 means it fully answers the question\n"
        'Return: {"faithfulness": <int>, "relevance": <int>, '
        '"unsupported_claims": [<short strings>]}'
    )
    resp = generate([{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": prompt}],
                    client=client, max_tokens=600, temperature=0.0)
    tokens = (resp.prompt_tokens or 0) + (resp.completion_tokens or 0)

    match = re.search(r"\{.*\}", resp.text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
        result = {
            "faithfulness": int(data["faithfulness"]),
            "relevance": int(data["relevance"]),
            "unsupported_claims": list(data.get("unsupported_claims", [])),
        }
        if not (1 <= result["faithfulness"] <= 5 and 1 <= result["relevance"] <= 5):
            raise ValueError
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        result = {"faithfulness": None, "relevance": None, "unsupported_claims": [],
                  "parse_error": resp.text[:200]}
    return result, tokens


# ---------- helpers ----------

def load_questions(limit: int | None) -> list[dict]:
    if not QUESTIONS_FILE.exists():
        sys.exit(f"{QUESTIONS_FILE} not found.")
    with open(QUESTIONS_FILE, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question", "").strip()]
    for r in rows:
        if r["expected"] not in ("answer", "refuse"):
            sys.exit(f"{r['qid']}: expected must be 'answer' or 'refuse'")
    return rows[:limit] if limit else rows


def load_corpus() -> dict[str, dict]:
    with open(CORPUS_FILE, encoding="utf-8") as f:
        return {rec["id"]: rec for rec in map(json.loads, f)}


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 3) if values else None


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation of the RAG system.")
    parser.add_argument("--api", help="test through the running API, e.g. http://127.0.0.1:8000")
    parser.add_argument("--judge", action="store_true", help="also grade answers with an LLM judge")
    parser.add_argument("--tpm", type=int, default=8000,
                        help="provider tokens-per-minute limit (0 = no waiting)")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--support-threshold", type=float, default=SUPPORT_THRESHOLD)
    args = parser.parse_args()

    questions = load_questions(args.limit)
    corpus = load_corpus()
    runner = ApiRunner(args.api) if args.api else InProcessRunner()
    mode = f"api ({args.api})" if args.api else "in-process"

    judge_client = None
    if args.judge:
        judge_client = runner.llm or get_llm_client()

    budget = TokenBudget(args.tpm)
    results = []
    print(f"Running {len(questions)} questions, mode: {mode}\n")

    for row in questions:
        budget.wait(EST_TOKENS_PER_CALL)
        res = runner.ask(row["question"])
        budget.record(res.get("prompt_tokens", 0) + res.get("completion_tokens", 0))

        rec = {"qid": row["qid"], "category": row["category"], "expected": row["expected"],
               "question": row["question"], **res}

        if "error" in res:
            rec["behavior_ok"] = False
            print(f"{row['qid']} ERROR {res['error']}: {res['message'][:80]}")
            results.append(rec)
            continue

        refused = is_refusal(res["answer"], res["llm_called"])
        rec["refused"] = refused
        rec["behavior_ok"] = refused == (row["expected"] == "refuse")

        if not refused:
            sources = [dict(s, text=corpus[s["doc_id"]]["text"])
                       for s in res["sources"] if s["doc_id"] in corpus]
            cited, invalid = find_citations(res["answer"], len(res["sources"]))
            ground, mean_sim, unsupported = groundedness(
                runner.embedder, res["answer"], [s["text"] for s in sources],
                args.support_threshold)
            rec.update(
                n_citations=len(cited),
                invalid_citations=len(invalid),
                groundedness=round(ground, 3) if ground is not None else None,
                mean_support=round(mean_sim, 3) if mean_sim is not None else None,
                unsupported_sentences=unsupported,
                answer_relevance=round(answer_relevance(
                    runner.embedder, row["question"], res["answer"]), 3),
            )
            if judge_client:
                budget.wait(EST_TOKENS_PER_CALL)
                try:
                    verdict, tokens = llm_judge(judge_client, row["question"], res["answer"], sources)
                    budget.record(tokens)
                    rec.update(judge_faithfulness=verdict["faithfulness"],
                               judge_relevance=verdict["relevance"],
                               judge_unsupported=verdict["unsupported_claims"])
                except LLMError as e:
                    rec["judge_error"] = type(e).__name__

        status = "ok " if rec["behavior_ok"] else "BAD"
        detail = "refused" if refused else (
            f"cites {rec['n_citations']}, grounded {rec['groundedness']}")
        print(f"{row['qid']} [{status}] {row['category']:<13} {res['total_ms']:>8.0f} ms | {detail}")
        results.append(rec)

    # ---------- summary ----------
    ok = [r for r in results if "error" not in r]
    answered = [r for r in ok if not r["refused"]]
    totals = [r["total_ms"] for r in ok if r.get("llm_called")]
    by_category = {}
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == cat]
        by_category[cat] = round(sum(r["behavior_ok"] for r in rows) / len(rows), 3)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "questions": len(results),
        "errors": dict(Counter(r["error"] for r in results if "error" in r)),
        "behavior_accuracy": round(sum(r["behavior_ok"] for r in results) / len(results), 3),
        "behavior_by_category": by_category,
        "answered": len(answered),
        "citation_rate": mean([1.0 if r["n_citations"] else 0.0 for r in answered]),
        "answers_with_invalid_citations": sum(1 for r in answered if r["invalid_citations"]),
        "groundedness": mean([r["groundedness"] for r in answered]),
        "answer_relevance": mean([r["answer_relevance"] for r in answered]),
        "judge_faithfulness": mean([r.get("judge_faithfulness") for r in answered]),
        "judge_relevance": mean([r.get("judge_relevance") for r in answered]),
        "latency_ms": {
            "total_p50": pct(totals, 50), "total_p95": pct(totals, 95),
            "total_max": round(max(totals), 1) if totals else None,
            "retrieval_p50": pct([r["retrieval_ms"] for r in ok], 50),
            "generation_p50": pct([r["generation_ms"] for r in ok if r.get("llm_called")], 50),
            "client_p50": pct([r["client_ms"] for r in ok], 50),
        },
        "tokens_mean": {
            "prompt": mean([r["prompt_tokens"] for r in ok if r.get("llm_called")]),
            "completion": mean([r["completion_tokens"] for r in ok if r.get("llm_called")]),
        },
        "support_threshold": args.support_threshold,
    }

    print("\n=== Summary ===")
    print(f"Behavior accuracy:   {summary['behavior_accuracy']}  {by_category}")
    print(f"Citation rate:       {summary['citation_rate']} "
          f"| answers with invalid citations: {summary['answers_with_invalid_citations']}")
    print(f"Groundedness:        {summary['groundedness']} "
          f"(share of sentences with support >= {args.support_threshold})")
    print(f"Answer relevance:    {summary['answer_relevance']}")
    if args.judge:
        print(f"LLM judge:           faithfulness {summary['judge_faithfulness']} / 5, "
              f"relevance {summary['judge_relevance']} / 5")
    lat = summary["latency_ms"]
    print(f"Latency (LLM calls): p50 {lat['total_p50']} ms | p95 {lat['total_p95']} ms | "
          f"max {lat['total_max']} ms")
    print(f"  retrieval p50 {lat['retrieval_p50']} ms | generation p50 {lat['generation_p50']} ms")
    if summary["errors"]:
        print(f"Errors: {summary['errors']}")
    bad = [r["qid"] for r in results if not r["behavior_ok"]]
    if bad:
        print(f"Wrong behavior on: {', '.join(bad)}  (see {ANSWERS_FILE.name})")

    LOG_DIR.mkdir(exist_ok=True)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    columns = ["qid", "category", "expected", "refused", "behavior_ok", "n_citations",
               "invalid_citations", "groundedness", "answer_relevance", "judge_faithfulness",
               "judge_relevance", "retrieval_ms", "generation_ms", "total_ms", "client_ms",
               "prompt_tokens", "completion_tokens", "error"]
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {SUMMARY_FILE.name}, {RESULTS_FILE.name} and {ANSWERS_FILE.name} in logs/")


if __name__ == "__main__":
    main()