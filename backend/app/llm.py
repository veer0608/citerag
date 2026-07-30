"""The single answer function: (question, chunks) -> answer + citations.

Swappable behind one interface. If no API key is configured, it degrades
gracefully to an extractive fallback so the /query endpoint (and demos) still
work — retrieval and the whole eval harness never depend on this at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.retrieval import RetrievedChunk

_SYSTEM = (
    "You answer strictly from the provided context passages. Every claim must be "
    "grounded in a passage. Cite the passages you used by their [n] marker. If the "
    "answer is not in the context, say you don't know."
)


@dataclass
class Answer:
    text: str
    citations: list[dict]  # [{marker, chunk_id, document_id, page_number}]
    model: str


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


def _citations(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "marker": i,
            "chunk_id": str(c.chunk_id),
            "document_id": str(c.document_id),
            "page_number": c.page_number,
            "page_label": c.page_label,
            "page_citation": _cite_page(c),
        }
        for i, c in enumerate(chunks, start=1)
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
    citations = _citations(chunks)
    context = _format_context(chunks)

    if settings.openai_api_key:
        text, model = _answer_openai(question, context)
    elif settings.anthropic_api_key:
        text, model = _answer_anthropic(question, context)
    elif settings.local_llm_enabled:
        text, model = _answer_local(question, context)
    else:
        # No key: return the top passages verbatim so the endpoint still works and
        # citations are still exact. This is clearly labelled, not a silent stub.
        top = chunks[0].content if chunks else "(no chunks retrieved)"
        text = (
            "[no LLM key configured — returning top retrieved passage verbatim]\n\n"
            f"{top}"
        )
        model = "extractive-fallback"

    return Answer(text=text, citations=citations, model=model)
