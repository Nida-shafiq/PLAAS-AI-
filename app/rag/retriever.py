import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.rag.config import (
    FAISS_INDEX_PATH, META_JSONL_PATH,
    BI_ENCODER, RERANKER,
    DENSE_TOP_K, RERANK_TOP_K,
)

_index = None
_meta = None
_bi_encoder = None
_cross_encoder = None


def _load_resources():
    global _index, _meta, _bi_encoder, _cross_encoder
    if _index is not None:
        return
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(META_JSONL_PATH):
        raise RuntimeError(f"Index or metadata not found at {FAISS_INDEX_PATH} / {META_JSONL_PATH}")
    _index = faiss.read_index(FAISS_INDEX_PATH)
    with open(META_JSONL_PATH, "r", encoding="utf-8") as f:
        _meta = [json.loads(line.strip()) for line in f]
    _bi_encoder = SentenceTransformer(BI_ENCODER)
    _cross_encoder = CrossEncoder(RERANKER, device="cpu")


def retrieve_documents(query: str, dense_top_k: int = DENSE_TOP_K, rerank_top_k: int = RERANK_TOP_K) -> list[dict]:
    _load_resources()
    q_emb = _bi_encoder.encode(query, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb.reshape(1, -1))
    D, I = _index.search(q_emb.reshape(1, -1), dense_top_k)
    candidate_idxs = [int(i) for i in I[0] if i != -1]
    if not candidate_idxs:
        return []
    pairs = [(query, _meta[idx]["text"]) for idx in candidate_idxs]
    scores = _cross_encoder.predict(pairs)
    candidates = [{"score": float(s), **_meta[idx]} for idx, s in zip(candidate_idxs, scores)]
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    selected, seen_texts = [], set()
    for c in candidates:
        txt = c["text"].strip()
        if txt in seen_texts:
            continue
        seen_texts.add(txt)
        selected.append(c)
        if len(selected) >= rerank_top_k:
            break
    return selected
