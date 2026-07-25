"""Idempotent corpus seeding.

Default corpus: Berkshire Hathaway annual reports (2021-2023). One company across
three years — dense financial tables, footnotes, boilerplate repeated across years,
and exact numbers that must be retrieved from the right year's table. That makes
"did it retrieve the right number from the right year" an objective eval question.

Downloads any missing PDFs into backend/data/corpus/, then ingests every PDF found
there. Re-running is safe: downloads skip files already present, and ingestion
replaces a document's prior rows rather than duplicating them.

You can also just drop your own PDFs into backend/data/corpus/ and run this — it
ingests whatever is there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

# Make `app` importable when run as `python scripts/seed_corpus.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal  # noqa: E402
from app.ingest import ingest_pdf  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"

DEFAULT_CORPUS: dict[str, str] = {
    "berkshire_2023.pdf": "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
    "berkshire_2022.pdf": "https://www.berkshirehathaway.com/2022ar/2022ar.pdf",
    "berkshire_2021.pdf": "https://www.berkshirehathaway.com/2021ar/2021ar.pdf",
}

# Some hosts reject requests without a browser-ish UA.
_HEADERS = {"User-Agent": "CiteRAG/0.1 (educational RAG project; contact: local)"}


def download_defaults() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DEFAULT_CORPUS.items():
        dest = CORPUS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  have  {name}")
            continue
        print(f"  fetch {name} <- {url}")
        try:
            with httpx.stream("GET", url, headers=_HEADERS, follow_redirects=True, timeout=120) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for block in r.iter_bytes():
                        f.write(block)
        except Exception as e:  # noqa: BLE001
            if dest.exists():
                dest.unlink(missing_ok=True)
            print(f"  WARN could not download {name}: {e}")


def ingest_all() -> None:
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(
            f"No PDFs in {CORPUS_DIR}. Downloads may have failed — drop PDFs there "
            "manually and re-run."
        )
        return
    with SessionLocal() as session:
        for pdf in pdfs:
            print(f"  ingest {pdf.name} ...", end=" ", flush=True)
            result = ingest_pdf(session, pdf)
            note = (
                f"{result.n_empty_pages} empty pages (scanned? OCR needed)"
                if result.n_empty_pages
                else "ok"
            )
            print(f"{result.n_pages} pages -> {result.n_chunks} chunks [{note}]")


if __name__ == "__main__":
    print("Downloading default corpus (skip files already present)...")
    download_defaults()
    print("Ingesting corpus...")
    ingest_all()
    print("Done.")
