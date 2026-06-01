# AI-Powered-Legal-Chatbot
RAG-based multilingual legal advisory chatbot for Pakistan — built with FAISS HNSW, InLegalBERT, cross-encoder reranking, and Gemini 2.5 Flash. Covers Constitution, CrPC, and Police Order 2002.
# PLAAS AI — Pakistan Legal Awareness & Advisory System

An AI-powered legal chatbot that provides Pakistani citizens with 
simplified, accurate guidance on their legal rights — built using 
a custom RAG pipeline without relying on off-the-shelf frameworks.

## What it does
Ask any legal question in plain English and get answers grounded 
in actual Pakistani law — cited by source, page, and clause.

## Data Sources
- The Constitution of Pakistan
- Code of Criminal Procedure (CrPC)
- The Police Order, 2002

## How it works (RAG Pipeline)
1. PDFs extracted page-by-page using pdfplumber
2. Clause-aware chunking splits text at legal boundaries 
   (Section, Article, Subsection etc.) with 250-char overlap
3. Chunks embedded using InLegalBERT (law-ai/InLegalBERT) — 
   a legal-domain bi-encoder
4. Embeddings stored in a FAISS HNSW index for fast similarity search
5. At query time: dense retrieval (top 50) → cross-encoder reranking 
   (ms-marco-MiniLM) → top 7 results passed to LLM
6. Gemini 2.5 Flash generates clean, citation-grounded legal 
   explanations via a strict prompt
7. Streamlit frontend for user interaction

## Tech Stack
- Embeddings: sentence-transformers / InLegalBERT
- Vector Search: FAISS (HNSW index with L2 normalization)
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: Google Gemini 2.5 Flash API
- PDF Parsing: pdfplumber
- UI: Streamlit
- Language: Python
