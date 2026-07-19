"""Single place for every tunable. Change a model or chunk size here, nowhere else."""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# llama3.2:3b — smallest Ollama model with reliable tool calling; fits in 8GB RAM.
# Swap for llama3.1:8b / qwen3:8b on a bigger machine.
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "mxbai-embed-large"
OLLAMA_URL = "http://localhost:11434"

# ~1 paragraph of context per chunk; overlap so answers spanning a boundary
# appear whole in at least one chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# top-k chunks retrieved per query
RETRIEVAL_K = 4

# how many times the agent may rewrite a question after bad retrieval.
# 1, not 2: with a 3B grader a false "irrelevant" verdict is common, and every
# extra loop adds latency before an answer the guard can vet anyway.
MAX_REWRITES = 1

CHROMA_DIR = str(_ROOT / "chroma_db")
COLLECTION = "pdfs"
CHECKPOINT_DB = str(_ROOT / "memory.sqlite")
PAPERS_FILE = str(_ROOT / "papers.json")
