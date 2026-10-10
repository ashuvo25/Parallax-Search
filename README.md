# Parallax-Search

This project collects 5,000 recent AI papers from arXiv and turns them into a clean text dataset. Each paper is stored as one document containing its title and abstract.

## Project structure

```
Internship-project-Parallax-Labs/
├── data/
│   ├── raw/                  PDFs, one per paper
│   │   └── rejected/         papers removed by the language filter
│   └── processed/
│       └── clean_corpus.jsonl
├── scripts/
│   ├── verify_env.py
│   ├── fetch_arxiv.py
│   └── build_corpus.py
├── src/
│   ├── __init__.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

The `data/` folder, `.env` and `venv/` are in `.gitignore`, so the data is not in the repository. Run the steps below to generate it.

## Setup

Python 3.10 or newer is required. The scripts were written and run on Windows.

Create and activate a virtual environment from the project root:

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell
venv\Scripts\activate.bat        # CMD
source venv/bin/activate         # macOS / Linux
```

Install PyTorch first. For a CUDA build, get the install command from https://pytorch.org/get-started/locally/. Otherwise:

```bash
pip install torch
```

Then install the rest:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the pipeline

Run these from the project root with the virtual environment active, in this order.

### 1. Check the environment

```bash
python scripts/verify_env.py
```

Prints the PyTorch version and whether CUDA is available. A CPU-only machine will print "CUDA is not available", which is fine for this stage.

### 2. Download the papers

```bash
python scripts/fetch_arxiv.py
```

Uses the arXiv API to get the 5,000 most recent papers from cs.AI, cs.CL and cs.LG. For each paper it creates a one-page PDF with the title and abstract and saves it to `data/raw/`, named after the paper title. Characters that Windows does not allow in file names are removed, and long titles are cut to 150 characters. If two papers have the same title, the arXiv ID is added to the second file name.

This takes a few minutes. Files that already exist are skipped, so the script can be run again if it stops.

### 3. Run the tests

```bash
python -m pytest tests -v
```

Tests the cleaning functions before they are used on the data. All tests should pass before moving on.

### 4. Clean the PDFs

Back up the raw data first, since this step overwrites the files:

```powershell
Copy-Item data\raw data\raw_backup -Recurse
```

Then run:

```bash
python src/preprocessing.py
```

Each PDF in `data/raw/` is read, cleaned and saved again under the same name. Papers that are not in English are moved to `data/raw/rejected/`. A summary of how many files were cleaned, rejected or failed is printed at the end.

### 5. Build the dataset

```bash
python scripts/build_corpus.py
```

Reads the cleaned PDFs, checks each document and writes the final dataset to `data/processed/clean_corpus.jsonl`.

## Text cleaning

The functions are in `src/preprocessing.py` and are applied in this order:

1. `strip_html` removes HTML tags and decodes entities such as `&amp;`. It only parses the text as HTML when it finds a real tag like `<p>` or `<br>`, so math in abstracts such as `x<y` is left alone.
2. `normalize_unicode` applies NFKC normalization, which turns ligatures like `ﬁ` into `fi` and non-breaking spaces into normal spaces. It also removes zero-width and control characters.
3. `clean_whitespace` collapses line breaks, tabs and repeated spaces into single spaces.
4. `detect_language` identifies the language with langdetect. Only English documents are kept.

`clean_text` runs the first three steps together.

## Dataset

The final dataset is at:

```
data/processed/clean_corpus.jsonl
```

It has one JSON object per line, one line per paper:

| Field           | Description                                          |
| --------------- | ---------------------------------------------------- |
| `id`          | 12-character hash of the text, the same on every run |
| `title`       | paper title                                          |
| `abstract`    | paper abstract                                       |
| `text`        | title and abstract together                          |
| `lang`        | detected language, always`en`                      |
| `n_words`     | number of words in`text`                           |
| `source_file` | name of the PDF in`data/raw/`                      |

Example:

```json
{"id": "3f9a1c2b7d4e", "title": "Example Paper Title", "abstract": "We propose a method that...", "text": "Example Paper Title We propose a method that...", "lang": "en", "n_words": 187, "source_file": "Example Paper Title.pdf"}
```

A document is left out of the dataset if it is empty, has fewer than 20 words, is not in English, would still change if cleaned again, or has the same text as a document already added. After writing the file, the script reads it back and checks that every line has the expected fields, that no ID appears twice and that the number of lines is correct.

To load it:

```python
import pandas as pd

df = pd.read_json("data/processed/clean_corpus.jsonl", lines=True)
```
