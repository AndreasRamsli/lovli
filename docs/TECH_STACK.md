# Lovli Tech Stack (2026)

This document outlines the architecture and technology choices for the Lovli legal RAG system.

## Core Architecture
Lovli is an LLM-powered legal information tool using a multi-stage Retrieval-Augmented Generation (RAG) pipeline to provide accurate Norwegian legal information based on Lovdata's public datasets (~735 current laws + regulations).

## Current Stack (MVP)

| Component | Choice | Description |
|-----------|--------|-------------|
| **Language** | Python 3.11+ | Primary language for data processing and RAG. |
| **RAG Framework** | [LangChain](https://www.langchain.com/) | Orchestrates retrieval, generation, and streaming. |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Evaluation, experiment tracking, and LLM-as-judge. |
| **Vector Store** | [Qdrant](https://qdrant.tech/) | Cloud-hosted, Docker-based, or in-memory vector database. |
| **Embedding Model** | [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Multilingual model with strong Norwegian support. Produces both dense and sparse vectors. |
| **LLM** | GLM-4 / Any | Via [OpenRouter](https://openrouter.ai/) API — supports 100+ models. |
| **Frontend** | [Streamlit](https://streamlit.io/) | Rapid UI development for the chat interface. |

## Planned Additions

| Component | Choice | Purpose |
|-----------|--------|---------|
| **Pipeline Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Multi-step retrieval pipeline with conditional routing and agent loops. Replaces `create_retrieval_chain` for complex flows. |
| **Cross-Encoder Reranker** | [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Multilingual reranker for rescoring retrieved candidates by actual relevance. |
| **Hybrid Search** | Qdrant sparse vectors + BGE-M3 sparse encoder | Combines semantic (dense) and keyword (sparse/BM25) search in a single query. |

## Data Pipeline

### Current
1. **Source**: HTML files from Lovdata bulk downloads (`/data/nl` and `/data/sf`).
2. **Parser**: Custom BeautifulSoup4 parser to extract semantic units (Articles).
3. **Chunking**: Semantic chunking based on legal structure (one chunk per Article).
4. **Indexing**: BGE-M3 dense embeddings stored in Qdrant.

### Planned (Multi-Stage)
1. **Source**: Full Lovdata bulk downloads — all 735+ current laws and central regulations.
2. **Parser**: Extended to extract full hierarchy (Law -> Chapter -> Section -> Subsection) plus cross-references between sections.
3. **Three-Tier Indexing**:
   - **Tier 0 — Law catalog**: Structured metadata + LLM-generated summaries for all 735 laws. Used for query routing. Not vectorized.
   - **Tier 1 — Chapter index**: Vectorized chapter-level summaries (~3,000-5,000 entries) for narrowing within a law.
   - **Tier 2 — Article index**: Full article text with hybrid search (dense + sparse vectors) and rich metadata (law, chapter, cross-references, Lovdata URLs).
4. **Indexing**: BGE-M3 dense + sparse embeddings, metadata-filtered collections in Qdrant.

## Retrieval Pipeline

### Current
Single-pass retrieval: embed question -> top-k similarity search -> stuff into prompt -> generate.

### Target Architecture
```
User Question
    |
    v
Query Analysis (classify, rewrite, extract refs)
    |
    +-- Direct lookup (if explicit section reference)
    |
    +-- Law Routing (LLM picks 1-3 candidate laws from catalog)
            |
            v
      Hybrid Search (dense + sparse, filtered by law)
            |
            v
      Reranking (cross-encoder, top 5)
            |
            v
      Cross-Reference Expansion (fetch referenced sections)
            |
            v
      Answer Generation (streaming, with inline citations)
```

## Infrastructure
- **Local Development**: Python environment, no local GPU required.
- **APIs**: OpenRouter (LLM), Qdrant Cloud (Vector DB).
- **Deployment**: Streamlit Cloud or Hugging Face Spaces.
- **Evaluation**: LangSmith for per-stage metrics and experiment comparison.

## Why These Choices?

- **No Local GPU**: By using hosted APIs (OpenRouter) and lightweight local models (BGE-M3, bge-reranker), we bypass the need for expensive hardware.
- **Model Flexibility**: OpenRouter provides access to 100+ models through a single API, allowing easy model switching and benchmarking.
- **Norwegian Support**: BGE-M3 and bge-reranker-v2-m3 are multilingual models with strong Norwegian performance.
- **Hybrid Search**: Legal text requires both semantic understanding ("tenant rights" ~ "leietakers rettigheter") and exact matching ("§ 3-5"). BGE-M3's dual dense+sparse output makes this seamless in Qdrant.
- **LangGraph over simple chains**: Multi-stage retrieval with conditional routing (skip routing for direct lookups, add cross-reference expansion when needed) requires a graph-based orchestrator, not a linear chain.
- **Evaluation-Driven**: Every architectural change is measured via LangSmith experiments. We only keep changes that improve eval scores.
