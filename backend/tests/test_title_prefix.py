"""chunk_title_prefix_enabled: the title reaches the index text and the re-ranker,
never the stored content."""
from app import retrieval
from app.config import settings
from app.ingest import with_title
from app.retrieval import RetrievedChunk


def test_with_title_off_by_default():
    assert settings.chunk_title_prefix_enabled is False
    assert with_title("ITC Q1 FY27 results", "Excise duty 10035.63") == "Excise duty 10035.63"


def test_with_title_on(monkeypatch):
    monkeypatch.setattr(settings, "chunk_title_prefix_enabled", True)
    assert with_title("ITC Q1 FY27 results", "Excise duty 10035.63") == "ITC Q1 FY27 results\nExcise duty 10035.63"
    assert with_title(None, "x") == "x"


class _NamesTheCompany:
    """Stand-in cross-encoder: scores a passage 1 if it mentions ITC, else 0."""

    def predict(self, pairs):
        return [1.0 if "ITC" in passage else 0.0 for _, passage in pairs]


def _chunk(cid, doc):
    return RetrievedChunk(chunk_id=cid, document_id=doc, page_number=1, content="Excise duty 10035.63", score=0.0)


def test_rerank_sees_titles_but_content_is_untouched(monkeypatch):
    monkeypatch.setattr(retrieval, "_reranker", lambda: _NamesTheCompany())
    chunks = [_chunk("a", "doc-tcs"), _chunk("b", "doc-itc")]
    titles = {"doc-tcs": "TCS Q1 FY27 press release", "doc-itc": "ITC Q1 FY27 results"}

    top = retrieval._rerank("ITC excise duty", chunks, top_k=1, titles=titles)

    assert top[0].chunk_id == "b"
    assert top[0].content == "Excise duty 10035.63"


def test_rerank_without_titles_is_unchanged(monkeypatch):
    monkeypatch.setattr(retrieval, "_reranker", lambda: _NamesTheCompany())
    chunks = [_chunk("a", "doc-tcs"), _chunk("b", "doc-itc")]
    top = retrieval._rerank("ITC excise duty", chunks, top_k=2)
    assert [c.score for c in top] == [0.0, 0.0]
