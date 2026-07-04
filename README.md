# Q&A Chatbot

A collection of LangChain + Streamlit Q&A chatbot apps, built while following a GenAI course. Each subfolder is a standalone Streamlit app exploring a different LLM backend / retrieval pattern.

## Projects

| Folder | Description | Backend |
|---|---|---|
| `2-OpenAI Chatbot` | Simple chatbot; user supplies their own OpenAI key in the sidebar | OpenAI |
| `3-Ollama Chatbot` | Simple chatbot running fully locally | Ollama (local) |
| `4-RAG Document Q&A` | RAG over a fixed set of PDFs (`research_papers/`) using FAISS | Groq + OpenAI/HuggingFace embeddings |
| `4.1-RAG Q&A Conversation Along with pdf and Chat history` | Conversational RAG with chat history over user-uploaded PDFs, using Chroma | Groq |

### `2-OpenAI Chatbot`

Single-turn Q&A chatbot. The user pastes their own OpenAI API key into the sidebar, picks a model (`gpt-4o`, `gpt-4-turbo`, `gpt-4`), and adjusts `temperature`/`max_tokens`. LangSmith tracing is enabled via `LANGCHAIN_API_KEY`.

```bash
streamlit run "2-OpenAI Chatbot/main.py"
```

### `3-Ollama Chatbot`

Same single-turn Q&A pattern as above, but runs fully locally against Ollama (`mistral` or `phi3`, selectable in the sidebar) — no API key needed. Requires Ollama installed and the chosen model pulled locally; not deployable to a hosted environment like Streamlit Cloud since it needs a local Ollama server.

```bash
streamlit run "3-Ollama Chatbot/app.py"
```

### `4-RAG Document Q&A`

RAG over a fixed set of PDFs in `research_papers/` (currently `Attention.pdf`, `LLM.pdf`). Click "Document Embedding" to chunk the PDFs and build a FAISS index, then ask questions answered only from that context. Uses Groq (`llama-3.1-8b-instant`) for generation and OpenAI embeddings (`app_huggingfaceembedding.py` is a variant using HuggingFace embeddings instead).

Requires `GROQ_API_KEY` and `OPENAI_API_KEY` (or `HF_TOKEN` for the HuggingFace variant) in `.env`.

```bash
streamlit run "4-RAG Document Q&A/main.py"
```

### `4.1-RAG Q&A Conversation Along with pdf and Chat history`

The most complete app in this repo: conversational RAG over PDFs you upload yourself, with real chat history. It uses `create_history_aware_retriever` to reformulate follow-up questions (e.g. "what about the second one?") using prior turns before retrieving, then answers with `create_retrieval_chain`. Chat history is kept per `session_id` in `st.session_state`, so multiple conversations can run side by side. Uses Groq (`llama-3.3-70b-versatile`) for generation, entered as an API key directly in the app, and HuggingFace embeddings (`all-MiniLM-L6-v2`, needs `HF_TOKEN`) with a Chroma vector store rebuilt per upload.

```bash
streamlit run "4.1-RAG Q&A Conversation Along with pdf and Chat history/app.py"
```

Uploaded PDFs are written to temporary `temp_<filename>.pdf` files in that folder during processing; they're gitignored and safe to delete.

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`) with the keys you need:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
HF_TOKEN=...
LANGCHAIN_API_KEY=...
```

Not every app needs every key — check the relevant `app.py`/`main.py` for what it reads. Some apps (e.g. `2-OpenAI Chatbot`, `4.1-RAG Q&A Conversation...`) instead prompt for the API key directly in the Streamlit sidebar.

## Running an app

```bash
streamlit run "2-OpenAI Chatbot/main.py"
```

(swap in the path to whichever app you want to run)

## Notes

- `venv/` and `.env` are gitignored — never commit real API keys.
- `4-RAG Document Q&A/research_papers/` holds sample PDFs used by that app's demo; other stray/generated PDFs (e.g. `temp.pdf` from file uploads) are ignored.
