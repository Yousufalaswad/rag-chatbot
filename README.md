# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables natural language Q&A over uploaded PDF documents with source page citations.

## Live Demo
[Try it here](https://yousuf-rag-chatbot.streamlit.app)

## Tech Stack
- **LLM:** Llama 3.1 via Groq API
- **Embeddings:** Cohere embed-english-light-v3.0
- **Vector Store:** FAISS (local)
- **Framework:** LangChain
- **UI:** Streamlit

## Features
- Upload any PDF document
- Ask questions in natural language
- Answers grounded in document content only
- Source page citations on every answer

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to `.env` (see `.env.example`)
4. Run: `streamlit run app.py`