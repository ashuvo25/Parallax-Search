
"""Build one validated JSONL corpus from the cleaned PDFs in data/raw/."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_text, detect_language, read_pdf  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = ROOT / "data" / "processed" / "clean_corpus.jsonl"

MIN_WORDS = 20
TARGET_DOCS = 5000
REQUIRED_KEYS = {"id", "title", "abstract", "text", "lang", "n_words", "source_file"}


def build_record(pdf_path: Path) -> dict:
    text = clean_text(read_pdf(pdf_path))
    title = clean_text(pdf_path.stem)
    abstract = text[len(title):].strip() if text.startswith(title) else text
    return {
        "id": hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
        "title": title,
        "abstract": abstract,
        "text": text,
        "lang": detect_language(text),
        "n_words": len(text.split()),
        "source_file": pdf_path.name,
    }


def validate(rec: dict) -> str | None:
    """Return None if the record is valid, otherwise the reason it failed."""
    if not rec["text"]:
        return "empty"
    if rec["n_words"] < MIN_WORDS:
        return "too_short"
    if rec["lang"] != "en":
        return "wrong_lang"
    if clean_text(rec["text"]) != rec["text"]:
        return "not_clean"  # leftover HTML, odd Unicode or extra spaces
    return None


def check_output(expected_count: int) -> None:
    """Re-read the saved file and make sure it's correct."""
    ids = set()
    with open(OUT_FILE, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            assert set(rec) == REQUIRED_KEYS, f"Line {line_no}: wrong keys {set(rec)}"
            assert rec["id"] not in ids, f"Line {line_no}: duplicate id {rec['id']}"
            ids.add(rec["id"])
    assert len(ids) == expected_count, f"Expected {expected_count}, found {len(ids)}"


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    seen = set()

    tmp = OUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            try:
                rec = build_record(pdf_path)
            except Exception as e:
                print(f"Failed: {pdf_path.name} -> {e}")
                stats["failed"] += 1
                continue

            reason = validate(rec)
            if reason is None and rec["id"] in seen:
                reason = "duplicate"
            if reason:
                stats[reason] += 1
                continue

            seen.add(rec["id"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    tmp.replace(OUT_FILE)
    check_output(stats["kept"])

    print(dict(stats))
    print(f"Saved {stats['kept']} documents to {OUT_FILE}")
    if stats["kept"] < TARGET_DOCS:
        print(f"Warning: below the {TARGET_DOCS} document target.")


if __name__ == "__main__":
    main()