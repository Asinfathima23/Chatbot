# 📄 AI Document Q&A Chatbot (RAG)

Upload a PDF and ask questions about it in plain English. The app retrieves the most relevant parts of your document and uses an LLM to answer based on that content — a technique called **RAG (Retrieval-Augmented Generation)**.

## How it works

1. **Load** — the PDF is loaded and its text extracted.
2. **Chunk** — the text is split into overlapping ~1000-character chunks.
3. **Embed** — each chunk is converted into a vector embedding locally using `sentence-transformers` (free, no API needed).
4. **Store** — embeddings are stored in a FAISS vector index for fast search.
5. **Retrieve** — when you ask a question, the most relevant chunks are found.
6. **Generate** — those chunks + your question are sent to a free Groq-hosted LLM, which generates a grounded answer.

## Tech stack

- [Streamlit](https://streamlit.io/) — web UI
- [LangChain](https://www.langchain.com/) — RAG orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [Sentence-Transformers](https://www.sbert.net/) — local embeddings
- [Groq](https://groq.com/) — free, fast LLM inference

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/doc-qa-chatbot.git
cd doc-qa-chatbot
python -m venv venv
venv\Scripts\activate      # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys), then create a `.env` file: