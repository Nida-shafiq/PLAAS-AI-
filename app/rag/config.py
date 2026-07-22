import os
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/legal_hnsw.index")
META_JSONL_PATH = os.getenv("META_JSONL_PATH", "data/legal_corpus_meta.jsonl")

BI_ENCODER = "law-ai/InLegalBERT"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K = 50
RERANK_TOP_K = 7

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
