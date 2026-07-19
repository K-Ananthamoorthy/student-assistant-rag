"""Streamlit chat UI. Thin by design — all RAG/agent logic lives in rag/."""

import uuid

import streamlit as st

from rag.agent import ask, build_agent
from rag.ingest import clear_index, get_vectorstore, ingest_pdfs


@st.cache_resource
def get_graph():
    return build_agent()


def main():
    st.set_page_config(page_title="Student Assistant — Local RAG", page_icon="📚")
    st.title("📚 Chat with your PDFs — fully local")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Documents")
        n_indexed = len(get_vectorstore().get()["ids"])
        st.caption(f"{n_indexed} chunks indexed")

        pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process", disabled=not pdfs):
            with st.spinner("Extracting, chunking, embedding…"):
                n = ingest_pdfs(pdfs)
            st.success(f"Indexed {n} chunks")
            st.rerun()

        if st.button("Clear index"):
            clear_index()
            st.rerun()

        if st.button("New conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.rerun()

    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    if question := st.chat_input("Ask about your documents…"):
        st.chat_message("user").write(question)
        st.session_state.history.append(("user", question))
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            answer = ask(get_graph(), question, st.session_state.thread_id)
            st.write(answer)
        st.session_state.history.append(("assistant", answer))


if __name__ == "__main__":
    main()
