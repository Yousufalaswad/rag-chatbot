import streamlit as st
from rag_pipeline import build_vectorstore, build_qa_chain
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 RAG Chatbot")
st.caption("Upload a PDF and ask questions — answers cite source pages.")

groq_api_key = os.getenv("GROQ_API_KEY", "")
cohere_api_key = os.getenv("COHERE_API_KEY", "")

if not groq_api_key or not cohere_api_key:
    st.error("Missing API keys in .env file. Make sure GROQ_API_KEY and COHERE_API_KEY are both set.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_key = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Reading and indexing PDF..."):
            vs = build_vectorstore(uploaded_file, cohere_api_key)
            chain, retriever = build_qa_chain(vs, groq_api_key)
            st.session_state["chain"] = chain
            st.session_state["retriever"] = retriever
            st.session_state["file_key"] = file_key
            st.session_state["messages"] = []
        st.success(f"Indexed '{uploaded_file.name}'. Ask away!")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about the document..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state["chain"].invoke(question)

                source_docs = st.session_state["retriever"].invoke(question)
                pages = sorted(set(
                    doc.metadata.get("page", 0) + 1
                    for doc in source_docs
                ))
                citation = f"\n\n> **Source pages:** {', '.join(map(str, pages))}"
                full_answer = answer + citation

            st.markdown(full_answer)
        st.session_state["messages"].append({"role": "assistant", "content": full_answer})