"""Ingestion pipeline: PDF -> per-page text -> chunks -> Chroma vector store,
plus a structured "paper card" per document (title, topic, method, findings).

Kept separate from the agent so indexing (slow, once per document set) and
querying (fast, every question) are independent — the same split you'd make
in production between an ingestion job and a serving path.

Card extraction uses a labeled-lines format instead of JSON: with a 3B local
model, line parsing is far more reliable than schema-constrained output
(see the judge design note in rag/guards.py — same lesson).
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def _llm() -> ChatOllama:
    return ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)


def get_vectorstore() -> Chroma:
    """Persistent Chroma collection — survives app restarts."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def list_papers() -> dict:
    """Paper cards keyed by filename, persisted alongside the index."""
    path = Path(config.PAPERS_FILE)
    return json.loads(path.read_text()) if path.exists() else {}


_CARD_PROMPT = """Here is the beginning of a document a student uploaded for a research project.
Filename: {name}

{text}

Describe this document. Reply EXACTLY in this format, one item per line:
TITLE: <the document's title>
TOPIC: <what it is about, one short line>
METHOD: <the approach, methodology, or type of document, one short line>
FINDING: <a key point or finding>
FINDING: <a key point or finding>
FINDING: <a key point or finding>
Do not use the em dash character."""


def _paper_card(name: str, text: str) -> dict:
    """One structured summary per paper, parsed from labeled lines."""
    card = {"title": name, "topic": "", "method": "", "findings": []}
    try:
        reply = _llm().invoke(_CARD_PROMPT.format(name=name, text=text[:3500])).content
    except Exception:
        return card
    for line in reply.splitlines():
        label, _, value = line.strip().partition(":")
        value = value.strip()
        if not value:
            continue
        key = label.strip().upper()
        if key == "TITLE":
            card["title"] = value
        elif key == "TOPIC":
            card["topic"] = value
        elif key == "METHOD":
            card["method"] = value
        elif key == "FINDING" and len(card["findings"]) < 3:
            card["findings"].append(value)
    return card


def ingest_pdfs(files, on_progress=None) -> list[dict]:
    """Index uploaded PDFs and build a card per paper.

    `files` are file-like objects with a .name. Returns the new cards.
    `on_progress(message)` lets the UI narrate each stage.
    """
    say = on_progress or (lambda msg: None)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    vs = get_vectorstore()
    papers = list_papers()
    new_cards = []

    for f in files:
        name = Path(getattr(f, "name", "upload.pdf")).name
        say(f"Reading {name}...")
        pages = []
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(page_content=text, metadata={"source": name, "page": page_num})
                )
        if not pages:
            continue

        say(f"Indexing {name} ({len(pages)} pages)...")
        chunks = splitter.split_documents(pages)
        vs.add_documents(chunks)

        say(f"Summarising {name}...")
        card = _paper_card(name, "\n".join(p.page_content for p in pages[:4]))
        card["pages"] = len(pages)
        papers[name] = card
        new_cards.append(card)

    Path(config.PAPERS_FILE).write_text(json.dumps(papers, indent=1))
    return new_cards


_REVIEW_PROMPT = """You are helping a student draft a literature review for a project.
Here are the papers they collected, with key points:

{cards}

Write a short literature review draft in markdown with these sections:
## Overview (what this collection of papers covers, 2-3 sentences)
## What each paper contributes (one bullet per paper, name it)
## Open questions and gaps (2-3 bullets)
Do not use the em dash character. Do not invent papers that are not listed."""


def literature_review() -> str:
    """One-call literature review draft over the paper cards."""
    papers = list_papers()
    if not papers:
        return ""
    cards = "\n\n".join(
        f"Paper: {c['title']}\nTopic: {c['topic']}\nMethod: {c['method']}\n"
        + "\n".join(f"- {p}" for p in c["findings"])
        for c in papers.values()
    )
    return _llm().invoke(_REVIEW_PROMPT.format(cards=cards)).content


def clear_index() -> None:
    """Drop every indexed chunk and all paper cards (start a new project)."""
    vs = get_vectorstore()
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)
    Path(config.PAPERS_FILE).unlink(missing_ok=True)
