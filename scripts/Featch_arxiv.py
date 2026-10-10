import arxiv
import re
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 5000

# Windows Arial font, which supports accents and special characters
FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def safe_filename(title, max_len=150):
    title = " ".join(title.split())
    title = re.sub(r'[\\/:*?"<>|]', "", title)
    return title[:max_len].strip(" .")


def make_pdf(title, abstract, path):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("ArialUni", "", FONT)
    pdf.add_font("ArialUni", "B", FONT_BOLD)

    pdf.set_font("ArialUni", "B", 16)
    pdf.multi_cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("ArialUni", "", 11)
    pdf.multi_cell(0, 6, abstract, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(path))


client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
search = arxiv.Search(
    query="cat:cs.AI OR cat:cs.CL OR cat:cs.LG",
    max_results=TARGET,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

count = 0
for paper in client.results(search):
    title = " ".join(paper.title.split())
    abstract = " ".join(paper.summary.split())

    name = safe_filename(title)
    filename = f"{name}.pdf"
    if (OUT_DIR / filename).exists():
        filename = f"{name} ({paper.get_short_id()}).pdf"
    if (OUT_DIR / filename).exists():
        count += 1
        continue

    try:
        make_pdf(title, abstract, OUT_DIR / filename)
        count += 1
        if count % 100 == 0:
            print(f"Saved {count} PDFs...")
    except Exception as e:
        print(f"Failed: {title[:60]} -> {e}")

print(f"Done. {count} PDFs in {OUT_DIR}")