# Staples AI Assistant

A lightweight GenAI prototype demonstrating how retrieval-augmented generation (RAG)
can help Staples teams quickly access internal knowledge.

## About Data [Sample synthetic data is generated in the form of pdfs]
- Internal documents (policies, catalogs, FAQs, contracts) are almost always shared as PDFs.
- RAG pipelines are designed to extract, chunk, embed, and retrieve text from such unstructured sources.

## Features
- Upload internal PDFs
- Query using natural language
- LangChain + Chroma + OpenAI backend
- Streamlit chat interface

## Architecture
PDF → Chunk & Embed → Vector Store (Chroma) → RAG Chain → LLM → Streamlit UI
