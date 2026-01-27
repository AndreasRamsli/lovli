# Project Roadmap: Lovli MVP

This roadmap outlines the steps to build the first version of the Lovli legal assistant.

## Phase 1: Data Preparation (Current)
- [x] Download Lovdata public datasets (Laws and Regulations).
- [x] Define tech stack (LangChain, Qdrant, GLM-4.7, BGE-M3).
- [ ] Create XML parser for legal articles.
- [ ] Extract a subset of data (e.g., *Husleieloven*) for initial testing.

## Phase 2: RAG Pipeline
- [ ] Set up Qdrant Cloud collection.
- [ ] Implement indexing script (Embeddings -> Qdrant).
- [ ] Build basic LangChain retrieval chain.
- [ ] Implement prompt engineering with legal disclaimers.

## Phase 3: Frontend MVP
- [ ] Build Streamlit chat interface.
- [ ] Add source citations (links to Lovdata).
- [ ] Implement "Suggested Questions" based on common tenant issues.

## Phase 4: Validation & Polish
- [ ] Test with 20+ real-world tenant questions.
- [ ] Refine chunking strategy if retrieval is inaccurate.
- [ ] Deploy to Streamlit Cloud for private feedback.

## Future Ideas (Post-MVP)
- **GraphRAG**: Map dependencies between different laws.
- **Multi-Law Support**: Expand beyond *Husleieloven* to all consumer laws.
- **Proactive Alerts**: Notify users when relevant laws change.
- **Document Analysis**: Allow users to upload their own lease agreements for checking against the law.
- **Query Semantic Validation**: Use an LLM to analyze question semantics before retrieval to ensure the correct documents are fetched.
