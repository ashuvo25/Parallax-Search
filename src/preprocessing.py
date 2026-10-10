"""Clean each PDF in data/raw/ and overwrite it with a cleaned PDF of the same name."""
import html
import os
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langdetect import DetectorFactory, LangDetectException, detect
from pypdf import PdfReader

DetectorFactory.seed = 0

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REJECTED_DIR = RAW_DIR / "rejected"
TARGET_LANG = "en"

FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"

_HTML_TAG = re.compile(
    r"</?(p|br|div|span|a|b|i|em|strong|ul|ol|li|h[1-6]|sub|sup|"
    r"table|tr|td|th|script|style|img)\b[^>]*>",
    re.IGNORECASE,
)
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------- cleaning functions ----------

def strip_html(text: str) -> str:
    if not _HTML_TAG.search(text):
        return html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _CONTROL_CHARS.sub("", text)


def clean_whitespace(text: str) -> str:
    return " ".join(text.split())


def detect_language(text: str) -> str | None:
    if len(text) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def clean_text(text: str) -> str:
    return clean_whitespace(normalize_unicode(strip_html(text)))


# ---------- PDF read / write ----------

def read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def write_pdf(title: str, body: str, path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("ArialUni", "", FONT)
    pdf.add_font("ArialUni", "B", FONT_BOLD)

    pdf.set_font("ArialUni", "B", 16)
    pdf.multi_cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("ArialUni", "", 11)
    pdf.multi_cell(0, 6, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # write to a temp file first, then replace, so a crash never leaves a broken PDF
    tmp = path.with_suffix(".tmp")
    pdf.output(str(tmp))
    os.replace(tmp, path)


def process_pdf(pdf_path: Path) -> str:
    text = clean_text(read_pdf(pdf_path))
    if not text:
        return "empty"

    if detect_language(text) != TARGET_LANG:
        REJECTED_DIR.mkdir(exist_ok=True)
        os.replace(pdf_path, REJECTED_DIR / pdf_path.name)
        return "wrong_lang"

    # split title from abstract so the title stays bold
    title = clean_text(pdf_path.stem)
    body = text[len(title):].strip() if text.startswith(title) else text

    write_pdf(title, body, pdf_path)  # same folder, same name
    return "cleaned"


if __name__ == "__main__":
    stats = {"cleaned": 0, "empty": 0, "wrong_lang": 0, "failed": 0}

    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        try:
            stats[process_pdf(pdf_path)] += 1
        except Exception as e:
            print(f"Failed: {pdf_path.name} -> {e}")
            stats["failed"] += 1

    print(stats)