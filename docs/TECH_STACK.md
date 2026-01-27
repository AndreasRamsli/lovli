# Lovli Tech Stack (MVP 2026)

This document outlines the architecture and technology choices for the Lovli legal RAG MVP.

## Core Architecture
Lovli is an LLM-powered legal information tool using a Retrieval-Augmented Generation (RAG) pipeline to provide accurate Norwegian legal information based on Lovdata's public datasets.

| Component | Choice | Description |
|-----------|--------|-------------|
| **Language** | Python 3.11+ | Primary language for data processing and RAG. |
| **RAG Framework** | [LangChain](https://www.langchain.com/) | Orchestrates the retrieval and generation steps. |
| **Vector Store** | [Qdrant](https://qdrant.tech/) | Cloud-hosted, Docker-based, or in-memory vector database for semantic search. |
| **Embedding Model** | [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Multilingual model with strong Norwegian support (via Hugging Face API). |
| **LLM** | GLM-4 / Any | Via [OpenRouter](https://openrouter.ai/) API - supports 100+ models including GLM-4, GPT-4, Claude, etc. |
| **Frontend** | [Streamlit](https://streamlit.io/) | Rapid UI development for the chat interface. |

## Data Pipeline
1. **Source**: XML files from Lovdata (`/data/nl` and `/data/sf`).
2. **Parser**: Custom BeautifulSoup4 parser to extract semantic units (Articles, Sections).
3. **Chunking**: Semantic chunking based on legal structure (one chunk per Article).
4. **Indexing**: BGE-M3 embeddings stored in Qdrant.

## Infrastructure
- **Local Development**: Python environment, no local GPU required.
- **APIs**: OpenRouter (LLM), Hugging Face (Embeddings), Qdrant Cloud (Vector DB).
- **Deployment**: Streamlit Cloud or Hugging Face Spaces.

## Why these choices?
- **No Local GPU**: By using hosted APIs (OpenRouter, HF), we bypass the need for expensive local hardware.
- **Model Flexibility**: OpenRouter provides access to 100+ models through a single API, allowing easy model switching.
- **Norwegian Support**: BGE-M3 and GLM-4 are specifically chosen for their performance on Norwegian text.
- **Speed to MVP**: Streamlit and LangChain allow for a functional prototype in days rather than weeks.
