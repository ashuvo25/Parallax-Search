"""Named entity recognition with spaCy, plus evaluation against a hand-labeled set.

The default run adds a jargon filter and rules for known AI names. Use --baseline
to evaluate plain spaCy, so the two can be compared.
"""
import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = ROOT / "data" / "processed" / "clean_corpus.jsonl"
DOC_ENTITIES_FILE = ROOT / "data" / "processed" / "doc_entities.jsonl"
GOLD_FILE = ROOT / "eval" / "ner_gold.csv"   # committed: it's manual work
LOG_DIR = ROOT / "logs"

SPACY_MODEL = "en_core_web_sm"
LABELS = ("ORG", "PRODUCT", "PERSON", "GPE", "LANGUAGE")

# General terms and math names that spaCy wrongly treats as entities
JARGON = {
    "llm", "llms", "ai", "agi", "gpu", "gpus", "tpu", "cpu", "rl", "rlhf", "moe",
    "linear", "gaussian", "markov", "monte carlo", "fourier", "bayesian", "euclidean",
    "laplacian", "hessian", "jacobian", "lipschitz", "f1", "node", "nodes",
    "transformer", "transformers", "cnn", "cnns", "rnn", "lstm", "gnn", "gnns",
    "vlm", "vlms", "mllm", "mllms", "nlp", "sota", "api", "apis", "lora", "rag",
    "cot", "sft", "dpo", "ppo", "kl", "mse", "auc", "iou", "bleu", "rouge",
    "pde", "pdes", "ode", "odes", "eeg", "mri", "ct", "2d", "3d",
}

# Known AI names, tagged before spaCy's own model runs
MODEL_REGEX = (r"^(GPT|ChatGPT|Llama|LLaMA|Qwen|DeepSeek|Mistral|Mixtral|Gemma|"
               r"Claude|Gemini|BERT|RoBERTa|ViT|CLIP|T5|Whisper)[\w.\-]*$")
DATASETS = [
    "ImageNet", "MMLU", "GSM8K", "HumanEval", "MBPP", "COCO", "SQuAD", "GLUE",
    "SuperGLUE", "CIFAR-10", "CIFAR-100", "MNIST", "LibriSpeech", "WikiText",
    "HellaSwag", "TruthfulQA", "BIG-Bench", "SWE-bench",
]
ORGS = [
    "OpenAI", "Google", "Google DeepMind", "DeepMind", "Meta", "Microsoft",
    "Anthropic", "NVIDIA", "Nvidia", "Apple", "Amazon", "Alibaba", "Baidu",
    "Tencent", "ByteDance", "Hugging Face", "IBM", "Mistral AI", "Stanford",
    "MIT", "Carnegie Mellon",
]


# ---------- helpers ----------

def normalize(text: str) -> str:
    """Make 'The OpenAI.' and 'openai' count as the same entity."""
    text = text.strip().casefold()
    text = re.sub(r"^the\s+", "", text)
    return text.strip(" .,;:()[]\"'")


def load_corpus() -> list[dict]:
    with open(CORPUS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_nlp(use_rules: bool = True):
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer"])
    except OSError:
        sys.exit(f"spaCy model missing. Run: python -m spacy download {SPACY_MODEL}")

    if use_rules:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "PRODUCT", "pattern": [{"TEXT": {"REGEX": MODEL_REGEX}}]}]
        patterns += [{"label": "PRODUCT", "pattern": name} for name in DATASETS]
        patterns += [{"label": "ORG", "pattern": name} for name in ORGS]
        ruler.add_patterns(patterns)
    return nlp


def extract_entities(nlp, texts: list[str], use_rules: bool = True,
                     batch_size: int = 64) -> list[list[dict]]:
    """Entities per text, only the labels we care about, no duplicates."""
    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        seen, ents = set(), []
        for ent in doc.ents:
            if ent.label_ not in LABELS:
                continue
            text = ent.text.strip()
            key = (normalize(text), ent.label_)
            if len(key[0]) < 2 or key in seen:
                continue
            if use_rules and (key[0] in JARGON or text.islower()):
                continue  # jargon, or a lowercase word (real names are capitalized)
            seen.add(key)
            ents.append({"text": text, "label": ent.label_})
        results.append(ents)
    return results


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


# ---------- commands ----------

def cmd_sample(n: int, force: bool) -> None:
    if GOLD_FILE.exists() and not force:
        sys.exit(f"{GOLD_FILE} already exists. Use --force to overwrite "
                 f"(this deletes your labels).")

    records = load_corpus()
    random.seed(42)
    sample = random.sample(records, min(n, len(records)))

    GOLD_FILE.parent.mkdir(exist_ok=True)
    with open(GOLD_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "text", "gold_entities"])
        for rec in sample:
            writer.writerow([rec["id"], rec["title"], rec["text"], ""])
    print(f"Wrote {len(sample)} papers to {GOLD_FILE}. Fill in the gold_entities column.")


def read_gold() -> list[dict]:
    if not GOLD_FILE.exists():
        sys.exit(f"{GOLD_FILE} not found. Run: python src/ner.py sample")

    rows = []
    with open(GOLD_FILE, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("gold_entities") or "").strip()
            if not raw:
                continue
            gold = set()
            if raw.upper() != "NONE":
                for part in raw.split(";"):
                    if "|" not in part:
                        print(f"Skipping badly formatted entry in {row['id']}: {part!r}")
                        continue
                    text, label = part.rsplit("|", 1)
                    label = label.strip().upper()
                    if label not in LABELS:
                        print(f"Unknown label {label!r} in {row['id']}, skipped")
                        continue
                    gold.add((normalize(text), label))
            rows.append({"id": row["id"], "text": row["text"], "gold": gold})
    return rows


def cmd_evaluate(baseline: bool) -> None:
    rows = read_gold()
    if not rows:
        sys.exit("No labeled rows yet. Fill in gold_entities in the CSV first.")

    use_rules = not baseline
    nlp = load_nlp(use_rules)
    predictions = extract_entities(nlp, [r["text"] for r in rows], use_rules)

    strict, span = Counter(), Counter()
    per_label = defaultdict(Counter)
    missed, wrong = [], []

    for row, ents in zip(rows, predictions):
        gold = row["gold"]
        pred = {(normalize(e["text"]), e["label"]) for e in ents}

        tp, fp, fn = gold & pred, pred - gold, gold - pred
        strict.update(tp=len(tp), fp=len(fp), fn=len(fn))
        for _, label in tp:
            per_label[label]["tp"] += 1
        for _, label in fp:
            per_label[label]["fp"] += 1
        for _, label in fn:
            per_label[label]["fn"] += 1

        gold_spans, pred_spans = {t for t, _ in gold}, {t for t, _ in pred}
        span.update(tp=len(gold_spans & pred_spans),
                    fp=len(pred_spans - gold_spans),
                    fn=len(gold_spans - pred_spans))

        missed += [f"{t} ({l})" for t, l in fn]
        wrong += [f"{t} ({l})" for t, l in fp]

    mode = "baseline" if baseline else "with_rules"
    report = {
        "model": SPACY_MODEL,
        "mode": mode,
        "labeled_papers": len(rows),
        "strict": prf(strict["tp"], strict["fp"], strict["fn"]),
        "span_only": prf(span["tp"], span["fp"], span["fn"]),
        "per_label": {l: prf(c["tp"], c["fp"], c["fn"]) for l, c in sorted(per_label.items())},
        "most_missed": Counter(missed).most_common(15),
        "most_wrong": Counter(wrong).most_common(15),
    }

    print(f"\nMode: {mode} | {len(rows)} hand-labeled papers\n")
    print(f"{'':12} {'Precision':>9} {'Recall':>7} {'F1':>6}")
    for name in ("strict", "span_only"):
        m = report[name]
        print(f"{name:12} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['f1']:>6.3f}")
    print("\nPer label (strict):")
    for label, m in report["per_label"].items():
        print(f"  {label:10} P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"F1 {m['f1']:.3f}  (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")
    print("\nMost missed:", ", ".join(x for x, _ in report["most_missed"][:8]))
    print("Most wrong: ", ", ".join(x for x, _ in report["most_wrong"][:8]))

    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / f"ner_eval_{mode}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out}")


def cmd_run() -> None:
    records = load_corpus()
    nlp = load_nlp(use_rules=True)

    start = time.perf_counter()
    all_ents = extract_entities(nlp, [r["text"] for r in records], use_rules=True)
    elapsed = time.perf_counter() - start

    label_counts, top = Counter(), defaultdict(Counter)
    with open(DOC_ENTITIES_FILE, "w", encoding="utf-8") as f:
        for rec, ents in zip(records, all_ents):
            for e in ents:
                label_counts[e["label"]] += 1
                top[e["label"]][e["text"]] += 1
            f.write(json.dumps({"id": rec["id"], "entities": ents}, ensure_ascii=False) + "\n")

    with_ents = sum(1 for e in all_ents if e)
    print(f"Processed {len(records)} papers in {elapsed:.1f}s "
          f"({len(records) / elapsed:.0f} papers/s)")
    print(f"Papers with at least one entity: {with_ents} ({100 * with_ents / len(records):.1f}%)")
    for label in LABELS:
        common = ", ".join(t for t, _ in top[label].most_common(5))
        print(f"  {label:9} {label_counts[label]:>5} found | top: {common}")
    print(f"Saved to {DOC_ENTITIES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="NER with spaCy.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="create the CSV to label by hand")
    p_sample.add_argument("-n", type=int, default=50)
    p_sample.add_argument("--force", action="store_true")

    p_eval = sub.add_parser("evaluate", help="compare spaCy with your labels")
    p_eval.add_argument("--baseline", action="store_true",
                        help="plain spaCy, without the jargon filter and rules")

    sub.add_parser("run", help="extract entities from all papers")

    args = parser.parse_args()
    if args.command == "sample":
        cmd_sample(args.n, args.force)
    elif args.command == "evaluate":
        cmd_evaluate(args.baseline)
    elif args.command == "run":
        cmd_run()


if __name__ == "__main__":
    main()