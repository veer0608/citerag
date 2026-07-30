"""Unit tests for /ingest path confinement (no DB / no PDF parsing).

The endpoint is unauthenticated, so the only thing standing between a caller and
"extract the text of any file on this server" is this resolution step.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.config import settings
from app.routers.ingest import resolve_corpus_path


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Point the corpus at a temp dir holding one real PDF and one non-PDF."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    (root / "notes.txt").write_text("not a pdf")
    (tmp_path / "secret.pdf").write_bytes(b"%PDF-1.4 outside the corpus")
    monkeypatch.setattr(settings, "corpus_dir", root)
    return root


def test_accepts_a_relative_path_inside_the_corpus(corpus):
    assert resolve_corpus_path("report.pdf") == (corpus / "report.pdf").resolve()


def test_accepts_the_absolute_path_of_a_corpus_file(corpus):
    absolute = str((corpus / "report.pdf").resolve())
    assert resolve_corpus_path(absolute) == (corpus / "report.pdf").resolve()


def test_rejects_parent_traversal(corpus):
    # The classic escape: resolving both sides before comparing is what stops it.
    with pytest.raises(HTTPException) as e:
        resolve_corpus_path("../secret.pdf")
    assert e.value.status_code == 400


def test_rejects_an_absolute_path_outside_the_corpus(corpus):
    outside = str((corpus.parent / "secret.pdf").resolve())
    with pytest.raises(HTTPException) as e:
        resolve_corpus_path(outside)
    assert e.value.status_code == 400


def test_rejects_non_pdf_even_inside_the_corpus(corpus):
    with pytest.raises(HTTPException) as e:
        resolve_corpus_path("notes.txt")
    assert e.value.status_code == 400


def test_missing_file_inside_the_corpus_is_a_404(corpus):
    with pytest.raises(HTTPException) as e:
        resolve_corpus_path("absent.pdf")
    assert e.value.status_code == 404
