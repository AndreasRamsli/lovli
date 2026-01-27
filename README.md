# Lovli - Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for Norwegian legal information, built with LangChain, Qdrant, and GLM-4.7.

## Overview

Lovli helps private users find information about Norwegian laws using natural language queries. The system uses legal documents from Lovdata (under NLOD 2.0 license) and provides answers with proper citations.

**⚠️ Disclaimer**: This tool provides legal information, not legal advice. For professional legal advice, please consult a qualified lawyer.

## Features

- Natural language queries about Norwegian laws
- Semantic search using BGE-M3 embeddings
- Answers with source citations
- Streamlit-based chat interface
- Focus on tenant rights (Husleieloven) for MVP

## Tech Stack

- **Python 3.11+**
- **LangChain** - RAG orchestration
- **Qdrant** - Vector database
- **BGE-M3** - Multilingual embeddings
- **GLM-4 / Any LLM** - Via OpenRouter API (supports 100+ models)
- **Streamlit** - Web UI
- **BeautifulSoup4** - XML parsing

## Setup

### Prerequisites

- Python 3.11 or higher
- Qdrant instance (Cloud, local Docker, or in-memory for testing)
- OpenRouter API key (get one at https://openrouter.ai/)
- (Optional) Hugging Face API token for embeddings

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
├── data/                    # Raw XML files from Lovdata
│   ├── nl/                  # Norwegian laws
│   └── sf/                  # Regulations
├── docs/                    # Documentation
├── src/
│   └── lovli/
│       ├── __init__.py
│       ├── config.py        # Settings and configuration
│       ├── parser.py         # XML parsing logic
│       ├── indexer.py        # Vector indexing
│       └── chain.py          # LangChain RAG pipeline
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

## License

MIT License

## Acknowledgments

- Lovdata for providing free access to legal data under NLOD 2.0
- OpenRouter for unified LLM API access
- BAAI for BGE-M3 embeddings

## Roadmap

See [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) for planned features and improvements.
