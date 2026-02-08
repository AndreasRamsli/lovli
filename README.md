# Lovli - Legal RAG Assistant

A multi-stage Retrieval-Augmented Generation (RAG) assistant for Norwegian legal information, built with LangChain, Qdrant, and BGE-M3.

## Overview

Lovli helps private users find information about Norwegian laws using natural language queries. The system uses legal documents from Lovdata (under NLOD 2.0 license) covering all 735+ current Norwegian laws, and provides answers with proper citations and source links.

**Warning**: This tool provides legal information, not legal advice. For professional legal advice, please consult a qualified lawyer.

## Features

- Natural language queries about Norwegian laws
- Multi-stage retrieval: query analysis -> law routing -> hybrid search -> reranking
- Hybrid search combining semantic (dense) and keyword (sparse) matching
- Answers with inline source citations and Lovdata links
- Real-time streaming responses
- Streamlit-based chat interface
- Evaluation via LangSmith with LLM-as-judge

## Architecture

```
User Question
    |
    v
Query Analysis (classify, rewrite, extract refs)
    |
    v
Law Routing (identify relevant laws from catalog)
    |
    v
Hybrid Search (dense + sparse vectors, metadata-filtered)
    |
    v
Reranking (cross-encoder rescoring)
    |
    v
Answer Generation (streaming, inline citations)
```

See [docs/RAG_CONCEPTS.md](docs/RAG_CONCEPTS.md) for a detailed explanation of each stage.

## Tech Stack

- **Python 3.11+**
- **LangChain / LangGraph** - RAG orchestration and multi-step pipeline
- **LangSmith** - Evaluation and observability
- **Qdrant** - Vector database with hybrid search (dense + sparse)
- **BGE-M3** - Multilingual embeddings (dense + sparse)
- **bge-reranker-v2-m3** - Cross-encoder reranker
- **GLM-4 / Any LLM** - Via OpenRouter API (supports 100+ models)
- **Streamlit** - Web UI
- **BeautifulSoup4** - HTML parsing

See [docs/TECH_STACK.md](docs/TECH_STACK.md) for detailed architecture decisions.

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

5. **Set up Qdrant** (choose one):
   
   **Option A: Qdrant Cloud**
   - Sign up at https://cloud.qdrant.io/
   - Create a cluster and get your URL and API key
   - Add to `.env`:
     ```
     QDRANT_URL=https://your-cluster.qdrant.io
     QDRANT_API_KEY=your_api_key
     ```

   **Option B: Local Docker**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   - Use `QDRANT_URL=http://localhost:6333` in `.env`

6. **Index legal documents**:
   ```bash
   python -m src.lovli.indexer  # (Script to be created)
   ```

7. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
lovli/
├── data/                    # Raw HTML files from Lovdata (gitignored)
│   ├── nl/                  # Norwegian laws
│   └── sf/                  # Regulations
├── docs/                    # Documentation
│   ├── PROJECT_ROADMAP.md   # Development roadmap and milestones
│   ├── TECH_STACK.md        # Architecture and technology choices
│   └── RAG_CONCEPTS.md      # Retrieval pipeline concepts explained
├── eval/                    # Evaluation datasets and results
├── scripts/                 # Utility scripts (eval, dataset upload)
├── src/
│   └── lovli/
│       ├── __init__.py
│       ├── config.py        # Settings and configuration
│       ├── parser.py        # HTML parsing logic
│       ├── indexer.py       # Vector indexing
│       └── chain.py         # RAG pipeline
├── app.py                   # Streamlit entry point
├── pyproject.toml           # Project metadata and dependencies
├── .env.example             # Template for API keys
└── README.md
```

## Usage

1. Start the Streamlit app: `streamlit run app.py`
2. Open your browser to the URL shown (typically http://localhost:8501)
3. Ask questions about Norwegian law in natural language
4. Review answers and check source citations

## Data Sources

- **Lovdata**: Norwegian legal data under NLOD 2.0 license
- Bulk downloads: `gjeldende-lover.tar.bz2` (laws) and `gjeldende-sentrale-forskrifter.tar.bz2` (regulations)
- Data files should be placed in `data/nl/` (laws) and `data/sf/` (regulations)

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Evaluation

We use **LangSmith** for comprehensive evaluation with experiment tracking and LLM-as-judge evaluation across each pipeline stage.

```bash
# One-time: upload dataset to LangSmith
python scripts/upload_dataset.py

# Run evaluation experiment
python scripts/eval_langsmith.py
```

Results are tracked in LangSmith with detailed metrics and experiment comparison.

## Roadmap

See [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) for the full development roadmap, covering:
- Phase 4: Retrieval quality (hybrid search, reranking, confidence gating)
- Phase 5: Extended parser and data enrichment
- Phase 6: Multi-stage retrieval pipeline with three-tier indexing
- Phase 7: Full-corpus validation and deployment

## License

MIT License

## Acknowledgments

- Lovdata for providing free access to legal data under NLOD 2.0
- OpenRouter for unified LLM API access
- BAAI for BGE-M3 embeddings and bge-reranker-v2-m3
