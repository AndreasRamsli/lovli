# Product Requirements Document (PRD)
## Legal Sparring Partner – Norwegian Law Chat System
**Version**: 1.0  
**Date**: February 2026  
**Status**: MVP-focused  

### 1. Executive Summary
Build a precise, fast, secure "sparring partner" chatbot for Norwegian law (Lovdata exports).  
Users debate hypotheticals, get exact citations, and explore provisions interactively.  
**Core innovation**: Agentic RAG (not naive RAG) – an intelligent agent that queries the structured corpus, pulls verbatim sections, reasons, and iterates.  
**Goal**: Outperform static RAG on citation accuracy, multi-hop reasoning, and user trust.

### 2. Objectives & Success Metrics (MVP)
- **Precision**: ≥95% citation accuracy; always returns verbatim text + source (law §X).
- **Speed**: ≤3s average response (MVP); ≤1.5s for simple queries.
- **Security**: 100% on-prem/self-hosted; GDPR-compliant; no user data used for training.
- **User Satisfaction**: ≥4.5/5 on helpfulness (internal beta with lawyers).
- **Scope**: Full Norwegian laws (~260 MB HTML exports) → focused MVP on 5–10 high-use laws (e.g., Husleieloven, etc.).

### 3. User Personas & Use Cases
**Primary**: Norwegian lawyers, law students, compliance officers, citizens.  
**Key Flows** (MVP):
1. "Debate this scenario under husleieloven §2-5"
2. "Compare §1-8 with new 2026 amendment"
3. "What are the exact requirements for notification in Kapittel 1?"
4. "Hypothetical: Landlord X does Y – is it discrimination per §1-8?"

### 4. Functional Requirements
**4.1 Core Agent Behavior**
- Always uses tools before answering.
- Tools: semantic_search, get_exact_section(citation), browse_hierarchy(chapter), check_amendments.
- Multi-turn memory + state (conversation context + retrieved sections).
- Output format: Verbatim quote → reasoning → citation → disclaimer.

**4.2 Data Layer**
- Preprocess all HTML files → structured nodes.
- Hybrid index: vector + keyword + rich metadata.

**4.3 Non-functional**
- Precise: Reranking + corrective retrieval.
- Fast: Caching, hybrid search, small router LLM.
- Secure: Docker/K8s deployment, encryption, audit logs, rate limiting.

### 5. Non-Functional Requirements
- **Precision**: Agentic loop + reranker + verbatim grounding.
- **Performance**: Sub-100ms retrieval; async processing.
- **Security & Compliance**: On-prem only; AES-256 at rest; user auth (JWT/OAuth); full audit trail; "Not legal advice" watermark.
- **Scalability**: Handle 100 concurrent users on mid-tier server ($30–60/mo).
- **Maintainability**: Ingestion pipeline auto-runs on law updates.

### 6. Assumptions & Risks
- Laws provided as HTML exports (not live API – Lovdata Pro API optional Phase 2).
- Norwegian embeddings available (bge-m3 confirmed excellent).
- Risk: Hallucination → mitigated by strict tool-first policy.
- Risk: Law updates → idempotent ingestion + versioning.

### 7. Out of Scope (MVP)
- Case law, preparatory works, full Lovdata API sync, multi-jurisdiction, voice, mobile app.