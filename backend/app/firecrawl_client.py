"""Thin client for the Firecrawl API — turns a URL, or a whole docs site, into
markdown pages ready for ingestion.

Kept separate from web_ingest.py so the network call is the only thing a test
needs to mock. Uses raw HTTP (httpx, already a dependency) rather than the
firecrawl-py SDK, since the REST surface used here is small.
"""
from __future__ import annotations

import time

import httpx

from app.config import settings

_BASE_URL = "https://api.firecrawl.dev/v1"


class FirecrawlError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    if not settings.firecrawl_api_key:
        raise FirecrawlError(
            "FIRECRAWL_API_KEY is not set — add it to backend/.env"
        )
    return {"Authorization": f"Bearer {settings.firecrawl_api_key}"}


def scrape(url: str) -> dict:
    """Scrape a single URL. Returns {"url", "title", "markdown"}."""
    resp = httpx.post(
        f"{_BASE_URL}/scrape",
        headers=_headers(),
        json={"url": url, "formats": ["markdown"]},
        timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise FirecrawlError(f"scrape failed for {url}: {body}")
    return _page(body["data"], fallback_url=url)


def crawl(
    url: str,
    *,
    limit: int = 50,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    poll_interval: float = 3.0,
    timeout: float = 600.0,
) -> list[dict]:
    """Crawl a site starting at `url`, up to `limit` pages.

    `include_paths`/`exclude_paths` are regexes matched against each candidate
    URL's path (Firecrawl's own crawl filters) — useful for docs sites that
    serve translated copies of every page under a locale prefix ("/es/...",
    "/fr/...": without excluding those, a small `limit` fills up with
    translations instead of covering the English tree).

    Firecrawl crawls run async: this starts the job, polls until it reports
    "completed" (raising FirecrawlError on "failed" or on exceeding `timeout`),
    then follows the result's pagination to collect every page. Returns a list
    of {"url", "title", "markdown"} dicts, one per crawled page.
    """
    payload: dict = {"url": url, "limit": limit, "scrapeOptions": {"formats": ["markdown"]}}
    if include_paths:
        payload["includePaths"] = include_paths
    if exclude_paths:
        payload["excludePaths"] = exclude_paths
    start = httpx.post(
        f"{_BASE_URL}/crawl",
        headers=_headers(),
        json=payload,
        timeout=30.0,
    )
    start.raise_for_status()
    job = start.json()
    if not job.get("success"):
        raise FirecrawlError(f"crawl failed to start for {url}: {job}")
    status_url = f"{_BASE_URL}/crawl/{job['id']}"

    deadline = time.monotonic() + timeout
    while True:
        resp = httpx.get(status_url, headers=_headers(), timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if status == "completed":
            break
        if status == "failed":
            raise FirecrawlError(f"crawl of {url} failed: {body}")
        if time.monotonic() > deadline:
            raise FirecrawlError(f"crawl of {url} did not finish within {timeout}s")
        time.sleep(poll_interval)

    pages = [_page(p, fallback_url=url) for p in body.get("data", [])]
    next_url = body.get("next")
    while next_url:
        resp = httpx.get(next_url, headers=_headers(), timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
        pages.extend(_page(p, fallback_url=url) for p in body.get("data", []))
        next_url = body.get("next")
    return pages


def _page(data: dict, *, fallback_url: str) -> dict:
    metadata = data.get("metadata") or {}
    source_url = metadata.get("sourceURL") or fallback_url
    return {
        "url": source_url,
        "title": metadata.get("title") or source_url,
        "markdown": data.get("markdown") or "",
    }
