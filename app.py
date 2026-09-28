import shutil

import streamlit as st

import rag

st.set_page_config(page_title="Chat with your documents", page_icon="📄")
st.title("📄 Chat with your documents (RAG)")

# ---- Sidebar: upload + index ----
with st.sidebar:
    st.header("1. Add documents")
    uploads = st.file_uploader(
        "PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Build index") and uploads:
        rag.DOCS_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (rag.DOCS_DIR / f.name).write_bytes(f.getbuffer())
        with st.spinner("Chunking and embedding..."):
            n = rag.build_index()
        st.session_state.pop("store", None)  # force reload of the new index
        st.success(f"Indexed {n} chunks.")

    if st.button("Clear index"):
        shutil.rmtree(rag.INDEX_DIR, ignore_errors=True)
        shutil.rmtree(rag.DOCS_DIR, ignore_errors=True)
        st.session_state.pop("store", None)
        st.session_state.messages = []
        st.info("Cleared.")

    st.header("2. Options")
    use_rerank = st.checkbox("Cross-encoder re-ranking", value=False)

# ---- Load index once per session ----
if "store" not in st.session_state:
    try:
        st.session_state.store = rag.load_index()
    except Exception:
        st.session_state.store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Chat ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask something about your documents"):
    if st.session_state.store is None:
        st.warning("Upload documents and click 'Build index' first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            index, chunks = st.session_state.store
            reply, sources = rag.answer(question, index, chunks, rerank=use_rerank)
        st.markdown(reply)
        with st.expander("Sources"):
            for n, s in enumerate(sources, start=1):
                st.markdown(f"**[{n}] {s['source']} (page {s['page']}, score {s['score']:.2f})**")
                st.caption(s["text"][:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": reply})