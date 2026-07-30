"""The single answer function: (question, chunks) -> answer + citations.

Swappable behind one interface. If no API key is configured, it degrades
gracefully to an extractive fallback so the /query endpoint (and demos) still
work — retrieval and the whole eval harness never depend on this at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.retrieval import RetrievedChunk

_SYSTEM = (
    "You answer strictly from the provided context passages. Every claim must be "
    "grounded in a passage, and you MUST cite the passage you used by writing its "
    "marker in square brackets — for example: Berkshire paid $3.3 billion in "
    "federal income taxes [2]. Cite every sentence that states a fact. If the "
    "answer is not in the context, say you don't know."
)

# A citation marker in the answer text: "[2]". Bounded to two digits — a longer
# bracketed number is a figure from the source, not a marker.
_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def used_markers(text: str, n_passages: int) -> list[int]:
    """The passage markers the answer actually cites, in order, de-duplicated.

    Markers outside 1..n_passages are dropped: a model that invents "[9]" against
    5 passages has cited nothing real, and passing it through would produce a
    citation pointing at no source.
    """
    seen: list[int] = []
    for raw in _MARKER_RE.findall(text):
        m = int(raw)
        if 1 <= m <= n_passages and m not in seen:
            seen.append(m)
    return seen


@dataclass
class Answer:
    text: str
    # Only the passages the answer actually cited — NOT the whole retrieved pool.
    citations: list[dict]
    model: str
    # True when the answer asserts something but cites no passage. The claim may
    # still be correct, but nothing in the response backs it up, so a caller should
    # treat it as unverified rather than sourced.
    uncited: bool = False


def _cite_page(chunk: RetrievedChunk) -> str:
    """How a page is described to a human: the printed label when we have one, with
    the physical PDF index alongside since that's what a viewer's page box wants."""
    if chunk.page_label:
        return f"page {chunk.page_label} (PDF page {chunk.page_number})"
    if chunk.page_number is not None:
        return f"PDF page {chunk.page_number}"
    return ""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page = _cite_page(c)
        blocks.append(f"[{i}]{f' ({page})' if page else ''} {c.content}")
    return "\n\n".join(blocks)


def _citations(chunks: list[RetrievedChunk], markers: list[int]) -> list[dict]:
    """Build citation records for just the markers the answer used.

    `markers` are 1-based positions into `chunks` (already range-checked), so the
    result is the passages actually cited rather than everything retrieved. The full
    retrieved pool is still returned separately by /query as `chunks`.
    """
    return [
        {
            "marker": m,
            "chunk_id": str(chunks[m - 1].chunk_id),
            "document_id": str(chunks[m - 1].document_id),
            "page_number": chunks[m - 1].page_number,
            "page_label": chunks[m - 1].page_label,
            "page_citation": _cite_page(chunks[m - 1]),
        }
        for m in markers
    ]


def _answer_openai(question: str, context: str) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    model = "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or "", model


def _answer_anthropic(question: str, context: str) -> tuple[str, str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    model = "claude-haiku-4-5-20251001"
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, model


@lru_cache
def _local_model():
    """Load the local instruct model once. Downloaded from HuggingFace on first
    use (no API key), then cached on disk. CPU-friendly small model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = settings.local_llm_model
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model


def _answer_local(question: str, context: str) -> tuple[str, str]:
    """Synthesize an answer with a small local LLM — free, no key, runs on CPU."""
    import torch

    tokenizer, model = _local_model()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # greedy: faster on CPU and deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, settings.local_llm_model


def answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
    context = _format_context(chunks)

    if settings.openai_api_key:
        text, model = _answer_openai(question, context)
    elif settings.anthropic_api_key:
        text, model = _answer_anthropic(question, context)
    elif settings.local_llm_enabled:
        text, model = _answer_local(question, context)
    else:
        # No key: return the top passage verbatim so the endpoint still works. It
        # quotes passage [1], so that marker is a true citation, not a stub.
        top = chunks[0].content if chunks else "(no chunks retrieved)"
        text = (
            "[no LLM key configured — returning top retrieved passage [1] verbatim]"
            f"\n\n{top}"
        )
        model = "extractive-fallback"

    # Citations are derived from the answer text, so they describe what was actually
    # used. A confident-looking answer that cites nothing is reported as uncited
    # rather than being dressed up with the whole retrieved pool.
    markers = used_markers(text, len(chunks))
    return Answer(
        text=text,
        citations=_citations(chunks, markers),
        model=model,
        uncited=bool(chunks) and not markers,
    )
