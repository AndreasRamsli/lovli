# Technical Architecture Specification
## Legal Sparring Partner – 2026 Best-Practice Stack

### 1. High-Level Architecture
Agentic RAG (LangGraph orchestration + LlamaIndex data layer)
User Query → Router LLM (cheap) → Agent (LangGraph)
↓
Tool Calls
┌──────────────────────┬──────────────────────┐
│ Semantic/Hybrid Search│ Exact Section Fetch  │ Browse Hierarchy
└──────────────────────┴──────────────────────┘
↓
Retrieved Nodes (verbatim + metadata)
↓
Synthesis LLM (Grok/Claude/Llama-3.1-70B) + Reranker
↓
Response with citations + sources
text### 2. Data Preprocessing Pipeline (Integrates Your Existing Code)
- **Input**: Your ~260 MB HTML files (Lovdata format with <header>, TOC, #kapittel-X, #§X IDs).
- **Step 1**: Reuse/enhance your parser (lxml/BeautifulSoup).
  - Extract metadata: legacyID, dokid, dateInForce, lastChangedBy, legalArea, etc.
  - Build hierarchy: Law → Kapittel → § → ledd/paragraph.
- **Step 2**: LlamaIndex HierarchicalNodeParser (or custom):
  - Parent nodes = full chapters/sections (summaries + metadata).
  - Child nodes = individual paragraphs (verbatim text).
  - Store cross-references as graph edges (Neo4j optional Phase 2).
- **Step 3**: Embed with `BAAI/bge-m3` (multilingual, top-tier for Norwegian + legal).
- **Output**: Stored in pgvector or Weaviate with metadata filters (law, chapter, effective_date, amendment_status).

**Estimated effort**: 1–2 days to integrate your code + add hierarchy.

### 3. Storage & Retrieval (Hybrid – Precision + Speed)
- **Primary**: PostgreSQL + pgvector (free, SQL + full-text + vector + metadata filters).
- **Alternative**: Weaviate (native hybrid + graph).
- **Retrieval**: Hybrid (vector 0.7 + BM25 0.3) + metadata filters + reranker (bge-reranker or Cohere).
- **Exact lookups**: SQL or metadata index for "§ 2-5 husleieloven".

### 4. Agent Framework
- **LangGraph** (stateful graphs, human-in-loop, retries, branching).
- Tools built with LlamaIndex query engines.
- Router decides: simple RAG vs full agent loop.

### 5. Backend & Deployment
- FastAPI + LangGraph API.
- Docker Compose (MVP) → Kubernetes (prod).
- LLM options: Grok API / Claude 3.5 / self-hosted Llama-3.1-70B (for max security).
- Caching: Redis (embeddings + frequent sections).

### 6. Security Controls
- All data on-prem or private VPC.
- End-to-end encryption.
- Session isolation per user.
- No logging of sensitive queries (configurable).
- Rate limiting + abuse detection.

### 7. Monitoring & Eval
- LangSmith / Phoenix for tracing.
- RAGAS + custom legal eval set (citation accuracy, completeness).