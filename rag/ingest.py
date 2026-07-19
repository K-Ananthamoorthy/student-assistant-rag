"""Ingestion pipeline: PDF bytes -> per-page text -> chunks -> Chroma vector store.

Kept separate from the agent so indexing (slow, once per document set) and
querying (fast, every question) are independent — the same split you'd make
in production between an ingestion job and a serving path.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def get_vectorstore() -> Chroma:
    """Persistent Chroma collection — survives app restarts."""
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def ingest_pdfs(files) -> int:
    """Index uploaded PDFs. `files` are file-like objects with a .name.

    Returns the number of chunks added.
    """
    # 1. Extract text per page, keeping source + page number as metadata
    #    so answers can cite where they came from.
    pages: list[Document] = []
    for f in files:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": getattr(f, "name", "upload"), "page": page_num},
                    )
                )

    # 2. Split into chunks. Recursive splitter tries paragraph -> line -> word
    #    boundaries in order, so chunks stay semantically whole.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        return 0

    # 3. Embed + store. Chroma persists to disk automatically.
    get_vectorstore().add_documents(chunks)
    return len(chunks)


def clear_index() -> None:
    """Drop every indexed chunk (start over with new documents)."""
    vs = get_vectorstore()
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)
