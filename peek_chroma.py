"""Peek inside the Chroma vector store. Run from the project root:

    uv run python peek_chroma.py
"""

from rag.ingest import get_vectorstore

vs = get_vectorstore()
data = vs.get(include=["documents", "metadatas"])   # vectors excluded by default

print("total chunks:", len(data["ids"]))
for i in range(min(5, len(data["ids"]))):
    md = data["metadatas"][i] or {}
    print(f"\n--- chunk {i}  [{md.get('source')} p.{md.get('page')}] ---")
    print((data["documents"][i] or "")[:300])
