"""Models, chunking, retrieval, and paths."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Swap for llama3.1:8b / qwen3:8b on a bigger machine.
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 4

# Rewrites allowed after bad retrieval.
MAX_REWRITES = 1

CHROMA_DIR = str(_ROOT / "chroma_db")
COLLECTION = "pdfs"
CHECKPOINT_DB = str(_ROOT / "memory.sqlite")
PAPERS_FILE = str(_ROOT / "papers.json")
