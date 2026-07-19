# Student Assistant — Local Agentic RAG over PDFs

Chat with your PDFs, fully offline. The LLM is not a single-shot pipeline here — it's the **reasoning engine of a LangGraph agent** that decides when to retrieve, grades its own retrieval, rewrites bad queries, and refuses to answer what the documents don't support.

**Stack:** LangChain 1.x · LangGraph · Ollama (llama3.2 + mxbai-embed-large) · Chroma · Streamlit · uv

## Architecture

```
START ──► agent (llama3.2 + retrieve tool)
            │
            ├─ no tool call (greeting / meta) ──────────────► END
            │
            ▼ tool call
         retrieve (Chroma, top-k=4, per-page citations)
            ▼
         grade: "can these chunks answer the question?"
            ├─ yes ──► generate (grounded, cites [file p.N])
            │             ▼
            │          grounding guard: unsupported claims ⇒ refuse
            │             ▼
            │            END
            └─ no ───► rewrite question ──► agent   (max 2 retries)

SQLite checkpointer: conversation memory persists across app restarts
```

## Design decisions

- **Agent, not chain.** A `ConversationalRetrievalChain` always retrieves, even for "hello". Here the LLM *decides* — retrieval is a tool it calls when needed. The graph makes the control flow explicit and inspectable.
- **Self-correction (reflection).** Retrieved chunks are graded for relevance; on a miss the agent rewrites the query and retries, capped at 2 rewrites so it can't loop forever — a reliability bound, not an afterthought.
- **Grounding guardrail.** After generation, an LLM judge checks every claim against the retrieved context. Unsupported answer ⇒ explicit refusal instead of a hallucination.
- **Persistent memory.** LangGraph's SQLite checkpointer stores each conversation thread durably — close the app, reopen, continue the thread.
- **Chunking: recursive, 1000 chars / 200 overlap.** Splits on paragraph → line → word boundaries so chunks stay semantically whole; overlap keeps boundary-straddling facts retrievable.
- **Chroma over FAISS-in-memory.** The index persists across restarts and carries per-page metadata for citations — still zero-server. (pgvector when the corpus needs concurrent writers; managed stores at real scale.)
- **Everything local.** Models run via Ollama: no API keys, no data leaves the machine. `rag/config.py` is the single place to swap models or tune parameters.

## Run it

```bash
# 1. Ollama + models (~2.7 GB)
brew install ollama          # or https://ollama.com
ollama serve                 # in its own terminal
ollama pull llama3.2:3b
ollama pull mxbai-embed-large

# 2. Python env (uv: https://docs.astral.sh/uv/)
uv sync

# 3. App
uv run streamlit run app.py
```

Upload PDFs → **Process** → ask questions. Answers cite sources like `[notes.pdf p.3]`.

## Evaluation

A golden Q&A set (`evals/golden.jsonl`) is scored by an LLM judge on **faithfulness** (answer grounded in retrieved context) and **correctness** (matches reference answer):

```bash
uv run python -m evals.run_eval
```

## Project layout

```
app.py              Streamlit chat UI (thin — UI only)
rag/config.py       every model name & tunable in one place
rag/ingest.py       PDFs → pages → chunks → Chroma
rag/agent.py        the LangGraph agent (nodes, edges, checkpointer)
rag/guards.py       grounding judge + refusal
evals/              golden set + LLM-as-judge harness
```

## License

MIT
