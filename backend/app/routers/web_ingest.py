"""POST /ingest/web — scrape a URL (or crawl a whole site) via Firecrawl and
ingest each page as its own document.

Unauthenticated, like /ingest — this is a corpus-seeding project, not a
multi-tenant service, so no confinement check is needed here the way
resolve_corpus_path() confines /ingest to a local directory.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_session
from app.firecrawl_client import FirecrawlError, crawl, scrape
from app.ingest import ingest_web_page

router = APIRouter(tags=["ingest"])


class WebIngestRequest(BaseModel):
    url: str
    # False: scrape just this URL. True: crawl the site starting here, up to
    # `limit` pages, and ingest each one as its own document.
    crawl: bool = False
    limit: int = 50
    # Regexes matched against each candidate URL's path (crawl only) — e.g. a
    # docs site with per-locale copies of every page needs exclude_paths=["^/es/", ...]
    # or the limit fills up with translations instead of English coverage.
    include_paths: list[str] = []
    exclude_paths: list[str] = []


class WebIngestResponse(BaseModel):
    documents_ingested: int
    n_chunks: int
    urls: list[str]


@router.post("/ingest/web", response_model=WebIngestResponse)
def ingest_web(req: WebIngestRequest, session: Session = Depends(get_session)) -> WebIngestResponse:
    try:
        pages = (
            crawl(
                req.url,
                limit=req.limit,
                include_paths=req.include_paths or None,
                exclude_paths=req.exclude_paths or None,
            )
            if req.crawl
            else [scrape(req.url)]
        )
    except FirecrawlError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    urls: list[str] = []
    n_chunks = 0
    for page in pages:
        if not page["markdown"].strip():
            continue
        try:
            result = ingest_web_page(session, page["url"], page["markdown"], title=page["title"])
        except ValueError:
            continue
        urls.append(page["url"])
        n_chunks += result.n_chunks

    return WebIngestResponse(documents_ingested=len(urls), n_chunks=n_chunks, urls=urls)
