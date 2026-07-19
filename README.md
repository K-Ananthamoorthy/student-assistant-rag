# Doc Companion — Local Agentic RAG for Your PDFs

Upload your PDFs, research papers, class notes, reports, or personal documents, and work with them like a partner: structured document cards, cited answers across documents, graduated safety guardrails. Fully offline, no API keys, nothing leaves your machine.

The LLM is not a single-shot pipeline here. It is the reasoning engine of a **LangGraph agent** that routes each question, decides when to retrieve, grades its own retrieval, rewrites bad queries, and answers behind graduated guardrails: verified, cautioned, or refused.

**Stack:** LangChain 1.x · LangGraph · Ollama (llama3.2 + mxbai-embed-large) · Chroma · Streamlit · uv

## Flow

1. **Upload gate:** the app opens on a single screen; nothing else exists until you upload PDFs and hit Analyse.
2. **Analyse:** each paper is extracted, chunked, embedded, and summarized into a **paper card** (title, topic, method, key findings) persisted alongside the index.
3. **Work:** chat with cited answers; the sidebar lists every document in the project.

## Architecture

```
START -> route --(small talk)-----> chitchat -> END
           |   --(corpus-level)---> overview -> END   (answers from paper cards)
           |
           v (specific question)
         agent --(no tool call)--> END
           |
           v (tool call)
        retrieve (Chroma, top-k=4) --> grade --(related)--> generate
                                         |                     |
                                         v (unrelated, <=1)    v
                                      rewrite -> agent    citation guard
                                                          grounding judge:
                                                          ok -> answer
                                                          unsure -> answer + caution
                                                          (exhausted + unrelated -> refuse)

SQLite checkpointer: conversation memory persists across app restarts
```

## Design decisions (the "why")

- **Agent, not chain.** A fixed retrieval chain retrieves for every message, even "hello". Here control flow is an explicit graph and the LLM reasons inside it.
- **Query routing: corpus-level vs passage-level.** "Summarize all my papers" is a known weakness of top-k retrieval, since no k chunks represent the whole corpus. Such questions route to an overview node that answers from the per-paper cards built at ingestion time. Specific questions go through retrieval.
- **Self-correction (reflection).** Retrieved chunks are graded for relevance; on a miss the agent rewrites the query and retries, capped at 1: with a 3B grader a false "irrelevant" is common, and every extra loop is latency before an answer the guard vets anyway.
- **Graduated guardrails after generation** (hard refusal had too many false positives with a 3B judge on long documents):
  - *Citations:* a deterministic regex guard verifies every cited `[file p.N]` tag exists verbatim in the context, repairing or dropping invented ones (a 3B model sometimes invents page numbers).
  - *Grounding:* an LLM judge compares the answer to the retrieved context; an unverified answer ships with a caution note instead of being replaced.
  - *Refusal* is reserved for the case where retrieval stayed unrelated even after the rewrite: the documents genuinely don't cover the topic.
- **Small-model engineering, measured not assumed:**
  - JSON-constrained structured output was unreliable for verdicts on llama3.2:3b (tested: defaulted to false). All judges use plain-text YES/NO or labeled-line formats instead.
  - The model calls its tool on every message once tools are bound, so routing is an explicit few-shot classifier (6/8 zero-shot, 8/8 few-shot on a routing testset).
- **Chunking: recursive, 1000 chars / 200 overlap.** Paragraph -> line -> word boundaries keep chunks semantically whole; overlap keeps boundary-straddling facts retrievable.
- **Chroma over in-memory FAISS.** Persistent index with per-page metadata for citations, still zero-server. pgvector when the corpus needs concurrent writers.
- **Persistent memory.** LangGraph's SQLite checkpointer stores each conversation thread durably.
- **Everything local.** Models run via Ollama. `rag/config.py` is the single place to swap models or tune parameters.

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

A sample PDF to try: `docs/sample_study_notes.pdf`.

## Evaluation

A golden Q&A set (`evals/golden.jsonl`) is scored by an LLM judge on **faithfulness** (answer grounded in retrieved context) and **correctness** (matches reference answer):

```bash
uv run python -m evals.run_eval
```

## Project layout

```
app.py              Streamlit UI: upload gate, document cards, sidebar, chat
rag/config.py       every model name and tunable in one place
rag/ingest.py       PDFs -> pages -> chunks -> Chroma, plus paper cards
rag/agent.py        the LangGraph agent (routing, nodes, edges, checkpointer)
rag/guards.py       grounding judge + citation guard + refusal
evals/              golden set + LLM-as-judge harness
```

## License

MIT
