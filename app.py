"""Streamlit chat UI. Thin by design, all RAG/agent logic lives in rag/.

Flow: upload gate -> Analyse -> brief summary -> chat.
The chat never appears until at least one document is indexed.
"""

import uuid

import streamlit as st

from rag.agent import ask, build_agent
from rag.ingest import clear_index, get_vectorstore, ingest_pdfs, summarize_index


@st.cache_resource
def get_graph():
    return build_agent()


def upload_gate():
    """Fullscreen first step: nothing else exists until documents are analysed."""
    st.title("📚 Student Assistant")
    st.subheader("Chat with your PDFs, fully local and private")
    st.write("Step 1 of 2: upload one or more PDF documents to get started.")

    pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Analyse documents", type="primary", disabled=not pdfs):
        with st.status("Analysing your documents...", expanded=True) as status:
            st.write("Extracting text and splitting into chunks...")
            n = ingest_pdfs(pdfs)
            st.write(f"Indexed {n} chunks. Writing a brief summary...")
            st.session_state.summary = summarize_index()
            status.update(label="Analysis complete", state="complete")
        st.rerun()
    st.stop()


def chat_page():
    st.title("📚 Chat with your PDFs")

    # Summary survives only the session; rebuild it after an app restart.
    if not st.session_state.get("summary"):
        with st.spinner("Summarising your documents..."):
            st.session_state.summary = summarize_index()
    st.info(f"**About your documents:** {st.session_state.summary}")

    with st.sidebar:
        st.header("Documents")
        st.caption(f"{len(get_vectorstore().get()['ids'])} chunks indexed")

        more = st.file_uploader("Add more PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Analyse", disabled=not more):
            with st.spinner("Analysing..."):
                ingest_pdfs(more)
                st.session_state.summary = summarize_index()
            st.rerun()

        if st.button("Start over (clear documents)"):
            clear_index()
            st.session_state.clear()
            st.rerun()

        if st.button("New conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    if question := st.chat_input("Ask about your documents..."):
        st.chat_message("user").write(question)
        st.session_state.history.append(("user", question))
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            answer = ask(get_graph(), question, st.session_state.thread_id)
            st.write(answer)
        st.session_state.history.append(("assistant", answer))


def main():
    st.set_page_config(page_title="Student Assistant | Local RAG", page_icon="📚")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []

    if len(get_vectorstore().get()["ids"]) == 0:
        upload_gate()
    chat_page()


if __name__ == "__main__":
    main()
