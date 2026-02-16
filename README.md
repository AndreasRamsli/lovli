# Lovli - Legal RAG Assistant

A multi-stage Retrieval-Augmented Generation (RAG) assistant for Norwegian legal information, built with LangChain, Qdrant, and BGE-M3.

## Overview

Lovli helps private users find information about Norwegian laws using natural language queries. The system uses legal documents from Lovdata (under NLOD 2.0 license) covering all 735+ current Norwegian laws, and provides answers with proper citations and source links.

**Warning**: This tool provides legal information, not legal advice. For professional legal advice, please consult a qualified lawyer.

## Features

- **Natural language queries** about Norwegian laws
- **Multi-stage retrieval**: query rewrite -> optional law routing -> hybrid search -> reranking -> generation
- **Hybrid search** combining semantic (dense) and keyword (sparse) matching using BGE-M3
- **Cross-encoder reranking** using `bge-reranker-v2-m3` for high precision
- **Provision-first context assembly**: editorial notes are attached per provision and normalized deterministically
- **Hierarchical parsing**: extracts Law -> Chapter -> Section structure
- **Cross-reference extraction**: captures links between different laws
- **Confidence gating**: avoids low-confidence answers when retrieval is weak
- **Conversation-aware**: rewrites follow-up questions using chat history
- **Law routing (optional)**: lightweight lexical + reranker routing narrows retrieval to likely laws
- **Staged fallback routing**: uncertainty first broadens to routed law sets, then optionally escalates to unfiltered retrieval
- **Law coherence filtering**: removes low-confidence singleton sources from non-dominant laws
- **Law-aware rank fusion**: combines CE score with routing/coherence/affinity signals for deterministic final ordering
- **Trust profiles**: reusable retrieval/reranking presets for evaluation and threshold sweeps
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

- `RETRIEVAL_K` and `RETRIEVAL_K_INITIAL` (final result size and over-retrieval before reranking)
- `RERANKER_CONFIDENCE_THRESHOLD`, `RERANKER_MIN_DOC_SCORE`, `RERANKER_MIN_SOURCES` (confidence and per-doc filtering)
- `RERANKER_AMBIGUITY_MIN_GAP`, `RERANKER_AMBIGUITY_TOP_SCORE_CEILING`, `RERANKER_AMBIGUITY_GATING_ENABLED` (ambiguity gating)
- `LAW_ROUTING_*` (optional law routing thresholds and uncertainty fallback behavior)
- `LAW_COHERENCE_*` (cross-law contamination guardrails after reranking)
- `LAW_RANK_FUSION_*` and `LAW_UNCERTAINTY_LAW_CAP_*` (deterministic post-rerank ranking controls)
- `EDITORIAL_NOTES_PER_PROVISION_CAP`, `EDITORIAL_NOTE_MAX_CHARS`, `EDITORIAL_V2_COMPAT_MODE` (editorial note payload controls)
- `TRUST_PROFILE_NAME` (preset defaults, e.g. `balanced_v1`, `strict_v1`)

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

### Retrieval tuning and diagnostics

```bash
# Sweep retrieval/reranker thresholds against eval/questions.jsonl
python scripts/sweep_retrieval_thresholds.py

# Analyze cross-law contamination and routing fallback behavior
python scripts/analyze_law_contamination.py

# Run regression gates by strictness tier (v1, v2, v3)
python scripts/check_regression_gates.py --gate-tier v2
```

## Current status (Feb 2026)

- Runtime now uses staged uncertainty fallback with explicit fallback-stage diagnostics.
- Confidence gating is calibrated on CE reranker scores; rank fusion only affects final ordering.
- Strict fallback behavior keeps stage-1 results when stage-2 unfiltered fallback is disabled.
- Sweep logic now mirrors runtime coherence + law-aware rank fusion + uncertainty law-cap behavior.
- Contamination analysis includes `hard_cluster_summary` output for mismatch/fallback-heavy clusters.
- Regression baselines now provide tiered gates (`gates`, `gates_v2`, `gates_v3`) and gate checker supports `--gate-tier`.
- Calibration diagnostics are emitted in sweep results (`calibration_bins`, `expected_calibration_error`).

## License

MIT License

## Acknowledgments

- Lovdata for providing free access to legal data under NLOD 2.0
- OpenRouter for unified LLM API access
- BAAI for BGE-M3 and bge-reranker models
