# PLAAS AI — Agentic Legal Research System

**Pakistani Legal Advisory & Assistance System** — an agentic AI system for researching Pakistani law, built with LangGraph, FAISS retrieval, and Gemini.

🔗 **Live API:** https://loyal-blessing-production.up.railway.app
📦 **Repo:** https://github.com/Nida-shafiq/PLAAS-AI-

---

## What it does

PLAAS AI answers questions about Pakistani law — the Constitution, the Code of Criminal Procedure (1898), and Police Order 2002 — by combining:

- **Local legal knowledge retrieval** (FAISS + InLegalBERT embeddings + cross-encoder reranking) over the actual source texts
- **Web search fallback** (DuckDuckGo) for questions outside the local corpus
- **An LLM agent (Gemini)** that decides which tool to use, synthesizes an answer, and appends a legal disclaimer
- **Session memory**, so follow-up questions ("what section was that again?") resolve correctly within a conversation


## Architecture
User query
│
▼
FastAPI endpoint (/chat)
│
▼
LangGraph StateGraph agent (Gemini flash-lite-latest)
│
├── faiss_legal_search → FAISS HNSW index (InLegalBERT embeddings)
│ → Cross-encoder rerank (ms-marco-MiniLM)
│ → Ranked, cited legal passages
│
├── web_legal_search → DuckDuckGo search (fallback for non-local queries)
│
└── legal_disclaimer → Standard legal disclaimer, appended to substantive answers
│
▼
Response + session history saved

# **Corpus:** Constitution of Pakistan, Code of Criminal Procedure 1898, Police Order 2002 — extracted with `pdfplumber`, clause-aware chunked, embedded with `law-ai/InLegalBERT`, indexed with FAISS HNSW, reranked with a cross-encoder at query time.

## Tech stack

- **Agent orchestration:** LangGraph, LangChain
- **LLM:** Google Gemini (`gemini-flash-lite-latest`)
- **Retrieval:** FAISS (HNSW index), `sentence-transformers`, InLegalBERT embeddings, cross-encoder reranking
- **Web search tool:** DuckDuckGo Search API
- **Backend:** FastAPI, Uvicorn
- **PDF extraction:** pdfplumber
- **Deployment:** Railway (Railpack build)

## API usage

**Health check**
```bash
curl https://loyal-blessing-production.up.railway.app/health
```

**Chat**
```bash
curl -X POST https://loyal-blessing-production.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo1", "message": "Can police arrest without a warrant in Pakistan?"}'
```

Response:
```json
{
  "session_id": "demo1",
  "response": "Yes, under Pakistani law, the CrPC allows..."
}
```

Send another message with the same `session_id` to continue the conversation with full context.

## Project structure
app/
├── agent/
│ ├── graph.py # LangGraph StateGraph, agent logic, tool binding
│ ├── state.py # Agent state schema
│ └── tools/
│ ├── faiss_tool.py # Local legal corpus search
│ ├── web_search_tool.py # DuckDuckGo fallback search
│ └── disclaimer_tool.py # Legal disclaimer
├── rag/
│ ├── config.py # Model names, paths, top-k settings
│ └── retriever.py # FAISS search + cross-encoder rerank (lazy-loaded)
├── api/
│ ├── main.py # FastAPI app, /health and /chat endpoints
│ └── session.py # In-memory session history
└── core/
└── logger.py # Request/response logging

data/
├── raw_pdfs/ # Source legal documents
├── legal_hnsw.index # FAISS index (committed for reproducibility)
└── legal_corpus_meta.jsonl # Chunk metadata (source, page, clause, text)

dataembeddings.py # One-time corpus build script (PDF → chunks → embeddings → FAISS)

## Running locally

```bash
git clone https://github.com/Nida-shafiq/PLAAS-AI-.git
cd PLAAS-AI-
pip install -r requirements.txt
```

Create a `.env` file:

GEMINI_API_KEY=your_key_here
FAISS_INDEX_PATH=data/legal_hnsw.index
META_JSONL_PATH=data/legal_corpus_meta.jsonl

If `data/legal_hnsw.index` and `data/legal_corpus_meta.jsonl` aren't present, rebuild the corpus from the source PDFs:
```bash
python dataembeddings.py
```

Run the API:
```bash
uvicorn app.api.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Can police enter my house without a warrant?"}'
```

## Notes

- Uses `gemini-flash-lite-latest` rather than a pinned model version, since Google rotates model availability frequently on the free tier — this alias always resolves to a current, accessible Flash-Lite model.
- Disclaimer: this system provides general legal information based on public legal texts and is not a substitute for professional legal advice.




