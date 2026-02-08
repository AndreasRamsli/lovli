# Project Roadmap: Lovli

This roadmap outlines the path from MVP to a robust, multi-law legal assistant for Norwegian law.

## Vision: A Legal Sparring Partner
Lovli is not a replacement for legal expertise. It is a sparring partner that helps
users reason through Norwegian law by grounding every response in sources.

**Core principles**
- Back up every claim with cited law text.
- Ask clarifying questions before answering ambiguous cases.
- Encourage verification by showing the law text inline and linking to Lovdata.
- Acknowledge limitations and uncertainty.

**Target users**
- Domain experts (e.g., NAV caseworkers) who know the laws but want a thinking partner.
- Everyday Norwegians who need to understand their rights (tenants, consumers, employees).

## Milestones (Outcome-Focused)
1. **It Works**: Answer 15/16 eval questions with correct citations.
2. **It's Useful**: A real user prefers asking Lovli over Googling for at least one question.
3. **It Sparrs**: Handles multi-turn follow-ups and asks clarifying questions.
4. **It's Trustworthy**: Displays relevant law text inline and avoids hallucinated sections.
5. **Others Want It**: At least one external tester returns to use it again.
6. **Beyond Husleieloven**: Add a second law without degrading quality.
7. **Full Corpus**: Cover all 735+ Norwegian laws with robust multi-stage retrieval.

---

## Phase 1: Data Preparation (Complete)
- [x] Download Lovdata public datasets (Laws and Regulations).
- [x] Define tech stack (LangChain, Qdrant, GLM-4.7, BGE-M3).
- [x] Create HTML parser for legal articles (BeautifulSoup4).
- [x] Extract Husleieloven for initial testing.

## Phase 2: Basic RAG Pipeline (Complete)
- [x] Set up Qdrant Cloud collection.
- [x] Implement indexing script (Embeddings -> Qdrant).
- [x] Build basic LangChain retrieval chain.
- [x] Implement real-time streaming responses.
- [x] Migrate evaluations to LangSmith with LLM-as-judge.

## Phase 3: Frontend MVP (In Progress)
- [x] Build Streamlit chat interface.
- [x] Display relevant law text inline in the conversation.
- [ ] Link to Lovdata for source verification.
- [ ] Ask clarifying questions before answering ambiguous queries.
- [ ] Implement "Suggested Questions" based on common legal issues.

## Phase 4: Retrieval Quality (Next)
Improve retrieval precision before scaling to more laws.
- [ ] **Hybrid search**: Add sparse vectors (BM25) alongside dense embeddings in Qdrant using BGE-M3's built-in sparse encoder. Catches exact section references (e.g., "§ 3-5") that pure semantic search misses.
- [ ] **Cross-encoder reranker**: Over-retrieve (k=15), then rerank with `bge-reranker-v2-m3` to select the top 5 by actual relevance. Drop-in improvement, measurable via LangSmith.
- [ ] **Confidence gating**: When reranker scores are low, respond with "Jeg fant ikke et klart svar" instead of guessing. Ask for clarification when retrieval is weak.
- [ ] **Conversation-aware retrieval**: Rewrite follow-up questions using chat history before retrieval (e.g., "Hva er fristen for det?" -> "Hva er fristen for husleieokening?").
- [ ] Expand eval set to 30+ questions covering edge cases and multi-section answers.

## Phase 5: Extended Parser & Enrichment
Enrich the data layer to support multi-law retrieval and cross-referencing.
- [ ] **Hierarchical extraction**: Extend parser to capture full structure (Law -> Chapter -> Section -> Subsection) with chapter-level metadata.
- [ ] **Cross-reference parsing**: Detect and store explicit references between sections (e.g., "jf. forbrukerkjopsloven § 15") as structured metadata.
- [ ] **Law-level summaries**: Generate a 2-3 sentence LLM summary for each of the 735 laws, describing scope and target audience. This becomes the routing index.
- [ ] **Metadata-rich payloads**: Store `law_id`, `law_title`, `law_short_name`, `chapter_id`, `chapter_title`, `cross_references`, and `url` on every indexed article.

## Phase 6: Multi-Stage Retrieval Pipeline
Build the robust, multi-stage pipeline for full-corpus retrieval.

### Three-Tier Indexing
- [ ] **Tier 0 — Law catalog**: Structured table of all 735 laws with summaries. Small enough to fit in a prompt for routing (~50k tokens).
- [ ] **Tier 1 — Chapter-level index**: ~3,000-5,000 vectorized chapter summaries for narrowing within a law.
- [ ] **Tier 2 — Article-level index**: Full article text with hybrid search (dense + sparse) and rich metadata filters.

### Retrieval Stages
- [ ] **Query analysis**: Classify query type (section lookup, legal question, concept explanation), extract explicit references, detect legal domain, rewrite follow-ups.
- [ ] **Law routing**: For broad questions, use LLM + law catalog to identify 1-3 candidate laws before retrieval.
- [ ] **Filtered hybrid search**: Search article index filtered to candidate laws, combining semantic and keyword matching.
- [ ] **Reranking**: Cross-encoder rescoring of candidates.
- [ ] **Cross-reference expansion**: Automatically fetch articles referenced by the top results (e.g., "jf. § 9-10").
- [ ] Migrate pipeline orchestration from `create_retrieval_chain` to **LangGraph** for multi-step control flow.

## Phase 7: Validation & Scale
- [ ] Expand eval set to 50-100 questions spanning 10-15 consumer-relevant laws.
- [ ] Test categories: single-law/single-section, single-law/multi-section, cross-law, direct lookups, follow-ups, out-of-scope.
- [ ] Measure each pipeline stage independently (routing accuracy, retrieval recall, reranker precision, answer quality).
- [ ] Ensure 0 hallucinated law sections in 50 test queries.
- [ ] Deploy to Streamlit Cloud for private feedback.
- [ ] Index all 735 laws and regulations from Lovdata bulk downloads.

## Future Ideas (Post Full-Corpus)
- **GraphRAG**: Map dependency graph between laws using parsed cross-references.
- **Proactive Alerts**: Notify users when relevant laws change.
- **Document Analysis**: Allow users to upload contracts/agreements for checking against the law.
- **Multi-language**: Support English queries about Norwegian law.
- **Agentic RAG**: Full agent loop where the LLM decides when to retrieve more, ask for clarification, or synthesize across multiple searches.
