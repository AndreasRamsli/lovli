# Lovli - Legal RAG Assistant

A multi-stage Retrieval-Augmented Generation (RAG) assistant for Norwegian legal information, built with LangChain, Qdrant, and BGE-M3.

## Overview

Lovli helps private users find information about Norwegian laws using natural language queries. The system uses legal documents from Lovdata (under NLOD 2.0 license) covering all 735+ current Norwegian laws, and provides answers with proper citations and source links.

**Warning**: This tool provides legal information, not legal advice. For professional legal advice, please consult a qualified lawyer.

## Features

- **Natural language queries** about Norwegian laws
- **Multi-stage retrieval**: query analysis -> hybrid search -> reranking -> generation
- **Hybrid search** combining semantic (dense) and keyword (sparse) matching using BGE-M3
- **Cross-encoder reranking** using `bge-reranker-v2-m3` for high precision
- **Adaptive editorial context**: provisions are prioritized, while editorial notes are included as supplemental context when budget and query intent indicate relevance
- **Hierarchical parsing**: extracts Law -> Chapter -> Section structure
- **Cross-reference extraction**: captures links between different laws
- **Confidence gating**: avoids low-confidence answers when retrieval is weak
- **Conversation-aware**: rewrites follow-up questions using chat history
- **Law-aware evaluation labels** via `expected_sources` (`law_id` + `article_id`) for stricter citation matching
- **Streamlit-based chat interface** with inline citations and source links
- **Evaluation via LangSmith** with LLM-as-judge metrics

## Tech Stack

- **Python 3.11+**
- **LangChain** - RAG orchestration
- **LangSmith** - Evaluation and observability
- **Qdrant** - Vector database with hybrid search support
- **BGE-M3** - Multilingual embeddings (dense + sparse)
- **bge-reranker-v2-m3** - Cross-encoder reranker
- **GLM-4 / Any LLM** - Via OpenRouter API
- **Streamlit** - Web UI
- **BeautifulSoup4** - HTML parsing with hierarchical extraction

## Setup

### Prerequisites

- Python 3.11 or higher
- Qdrant instance (Cloud, local Docker, or in-memory for testing)
- OpenRouter API key (get one at https://openrouter.ai/)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lovli
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

### Data Preparation & Indexing

Lovli requires legal data from Lovdata's bulk downloads.

1. **Download data**:
   - Download `gjeldende-lover.tar.bz2` from Lovdata.
   - Extract to `data/nl/`.

2. **Build the law catalog** (Tier 0):
   ```bash
   # Scans all laws and generates LLM summaries (requires API key)
   python scripts/build_catalog.py data/nl/
   ```

3. **Index legal documents**:
   ```bash
   # Index a specific law (e.g., Husleieloven)
   python scripts/index_laws.py data/nl/nl-19990326-017.xml --recreate
   
   # Index all laws in the directory
   python scripts/index_laws.py data/nl/
   ```

4. **Reindex validation (recommended after full rebuild)**:
   ```bash
   # Ensures metadata.doc_type exists on all points
   python scripts/validate_reindex.py --require-zero-missing

   # Optional: include retrieval smoke checks
   python scripts/validate_reindex.py --require-zero-missing --with-smoke
   ```

### Retrieval controls

Recent retrieval behavior can be tuned without code changes through `.env`:

- `RETRIEVAL_K_INITIAL` (over-retrieval before reranking)
- `RERANKER_MIN_DOC_SCORE` and `RERANKER_MIN_SOURCES` (per-doc filtering safety floor)
- `RERANKER_AMBIGUITY_MIN_GAP` and `RERANKER_AMBIGUITY_TOP_SCORE_CEILING` (ambiguity gating)
- `EDITORIAL_BASE_MAX_NOTES`, `EDITORIAL_MAX_NOTES`, `EDITORIAL_CONTEXT_BUDGET_RATIO`, `EDITORIAL_HISTORY_INTENT_BOOST` (adaptive editorial context budgeting)

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

```
lovli/
├── data/                    # Raw HTML files and generated catalog
├── docs/                    # Detailed documentation
├── eval/                    # Evaluation datasets and results
├── scripts/                 # CLI tools for indexing, cataloging, and eval
│   ├── index_laws.py        # Index laws into Qdrant
│   ├── build_catalog.py     # Generate law catalog with summaries
│   ├── eval_langsmith.py    # Run LangSmith evaluations
│   ├── validate_reindex.py  # Post-index metadata/retrieval validation
│   └── upload_dataset.py    # Upload eval questions to LangSmith
├── src/
│   └── lovli/
│       ├── parser.py        # Hierarchical HTML parsing
│       ├── indexer.py       # Vector indexing (dense + sparse)
│       ├── chain.py         # Multi-stage RAG pipeline
│       ├── catalog.py       # Catalog generation logic
│       ├── config.py        # Pydantic settings
│       └── utils.py         # Shared utilities
├── tests/                   # Unit and integration tests
├── app.py                   # Streamlit entry point
├── pyproject.toml           # Dependencies
└── README.md
```

## Evaluation

We use **LangSmith** for comprehensive evaluation. Results are tracked with metrics for retrieval relevance, citation accuracy, correctness, and groundedness.

```bash
# Run evaluation experiment
python scripts/eval_langsmith.py
```

## License

MIT License

## Acknowledgments

- Lovdata for providing free access to legal data under NLOD 2.0
- OpenRouter for unified LLM API access
- BAAI for BGE-M3 and bge-reranker models
