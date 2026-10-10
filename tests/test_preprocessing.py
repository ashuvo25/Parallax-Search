"""Unit tests for the cleaning functions in src/preprocessing.py."""
import sys
from pathlib import Path

import pytest

# make "src" importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (  # noqa: E402
    clean_text,
    clean_whitespace,
    detect_language,
    normalize_unicode,
    strip_html,
)


# ---------- strip_html ----------

def test_strip_html_removes_tags():
    result = strip_html("<p>Hello <b>world</b></p>")
    assert " ".join(result.split()) == "Hello world"


def test_strip_html_decodes_entities():
    assert strip_html("Tom &amp; Jerry") == "Tom & Jerry"


def test_strip_html_removes_script_and_style_content():
    result = strip_html("<p>Text</p><script>alert(1)</script><style>p{}</style>")
    assert "alert" not in result
    assert "p{}" not in result
    assert "Text" in result


def test_strip_html_keeps_math_symbols():
    # arXiv abstracts contain math like this; it must NOT be treated as HTML
    text = "We show x<y and z>w for all inputs."
    assert strip_html(text) == text


def test_strip_html_plain_text_unchanged():
    assert strip_html("Just plain text.") == "Just plain text."


# ---------- normalize_unicode ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("\ufb01ne", "fine"),          # "fi" ligature -> two letters
        ("a\u00a0b", "a b"),           # non-breaking space -> normal space
        ("co\u200bde", "code"),        # zero-width space removed
        ("a\x00b", "ab"),              # control character removed
        ("ＡＢＣ", "ABC"),              # full-width letters -> normal
        ("e\u0301", "\u00e9"),         # e + accent -> single é
    ],
)
def test_normalize_unicode(raw, expected):
    assert normalize_unicode(raw) == expected


def test_normalize_unicode_keeps_newlines_and_tabs():
    assert normalize_unicode("a\nb\tc") == "a\nb\tc"


# ---------- clean_whitespace ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  a   b  ", "a b"),
        ("a\n\tb\r\nc", "a b c"),
        ("", ""),
        ("   \n\t ", ""),
    ],
)
def test_clean_whitespace(raw, expected):
    assert clean_whitespace(raw) == expected


# ---------- detect_language ----------

def test_detect_language_english():
    text = "This paper proposes a new method for training large language models."
    assert detect_language(text) == "en"


def test_detect_language_french():
    text = "Cet article propose une nouvelle méthode pour entraîner des modèles de langage."
    assert detect_language(text) == "fr"


@pytest.mark.parametrize("text", ["", "Too short", "123456789 0123456789 012345"])
def test_detect_language_returns_none_when_unclear(text):
    assert detect_language(text) is None


# ---------- clean_text (full pipeline) ----------

def test_clean_text_full_pipeline():
    raw = "<p>De\ufb01ning   new\u00a0models</p>\n\n &amp; more"
    assert clean_text(raw) == "Defining new models & more"


@pytest.mark.parametrize(
    "raw",
    [
        "<p>Hello   <b>world</b></p>",
        "\ufb01ne\u00a0text\n\nhere",
        "Plain sentence.",
    ],
)
def test_clean_text_is_idempotent(raw):
    # cleaning already-clean text must not change it
    once = clean_text(raw)
    assert clean_text(once) == once