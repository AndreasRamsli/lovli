# MVP Development Roadmap
**Timeline**: 3–4 weeks (solo or small team)

### Phase 0: Preparation (2–3 days)
- Finalize 5 sample laws (including Husleieloven).
- Integrate + test your existing parser on full set.
- Set up repo: FastAPI + LangGraph + LlamaIndex + pgvector.

### Phase 1: Ingestion & Indexing (Week 1)
- Preprocess pipeline → hierarchical nodes.
- Embed + store in pgvector.
- Build 4 tools (search, get_exact, browse, amendments).
- Test retrieval accuracy on 50 sample queries.

### Phase 2: Agent Core (Week 2)
- LangGraph agent with memory + tool-calling.
- System prompt enforcing verbatim + citations.
- Basic chat UI (Gradio or Streamlit for MVP).

### Phase 3: Polish & Eval (Week 3)
- Reranking + corrective loop.
- Speed optimizations (caching, async).
- Security hardening + disclaimers.
- Internal beta with 5–10 users + feedback loop.

### Phase 4: Deploy & Monitor (Week 4)
- Docker deployment.
- Eval dashboard.
- Ready for production expansion.

### Resources Needed
- Server: 16 GB RAM + GPU optional (CPU fine for MVP).
- Budget: <$100/mo (self-hosted models + pgvector).
- Your existing parser code → critical accelerator.

### Success Criteria for MVP Handover
- Agent answers 20 complex legal hypotheticals with perfect citations.
- Average response <3s.
- Zero hallucinations on test set.
- Secure, documented, runnable locally.