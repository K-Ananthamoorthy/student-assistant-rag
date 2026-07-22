"""Ingestion: PDF -> per-page text -> chunks -> Chroma, plus a "paper card"
(title, topic, method, findings) per document."""

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
    """Persistent Chroma collection."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def list_papers() -> dict:
    """Paper cards keyed by filename."""
    path = Path(config.PAPERS_FILE)
    return json.loads(path.read_text()) if path.exists() else {}


_CARD_PROMPT = """Here is the beginning of a document the user uploaded (it may be a research paper, notes, a report, a book, or a personal document).
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
    """Build one document card by parsing the model's labeled-line reply."""
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
    """Index uploaded PDFs and build a card per paper. Returns the new cards.

    `files` are file-like objects with a .name; `on_progress(message)` reports
    each stage to the UI.
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


def clear_index() -> None:
    """Drop every indexed chunk and all paper cards."""
    vs = get_vectorstore()
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)
    Path(config.PAPERS_FILE).unlink(missing_ok=True)
