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

## Phase 3: Frontend MVP (Complete)
- [x] Build Streamlit chat interface.
- [x] Display relevant law text inline in the conversation.
- [x] Link to Lovdata for source verification.
- [x] Ask clarifying questions before answering ambiguous queries.
- [x] Implement "Suggested Questions" based on common legal issues.

## Phase 4: Retrieval Quality (Complete)
Improve retrieval precision before scaling to more laws.
- [x] **Hybrid search**: Add sparse vectors (BM25) alongside dense embeddings in Qdrant using BGE-M3's built-in sparse encoder. Catches exact section references (e.g., "§ 3-5") that pure semantic search misses.
- [x] **Cross-encoder reranker**: Over-retrieve (k=15), then rerank with `bge-reranker-v2-m3` to select the top 5 by actual relevance. Drop-in improvement, measurable via LangSmith.
- [x] **Confidence gating**: When reranker scores are low, respond with "Jeg fant ikke et klart svar" instead of guessing. Ask for clarification when retrieval is weak.
- [x] **Ambiguity gating**: Gate responses when top reranker scores are too close and low-confidence.
- [x] **Per-document filtering**: Drop weak reranked documents while enforcing a minimum source floor.
- [x] **Conversation-aware retrieval**: Rewrite follow-up questions using chat history before retrieval (e.g., "Hva er fristen for det?" -> "Hva er fristen for husleieøkning?").
- [x] **Deduplication**: Implemented document deduplication before reranking to improve efficiency and result quality.
- [x] **Multilingual support**: Optimized for Norwegian legal text using BGE-M3 and specific Lovdata parsing rules.
- [x] Expand eval set to 30+ questions covering edge cases and multi-section answers.

## Phase 5: Extended Parser & Enrichment (Complete)
Enrich the data layer to support multi-law retrieval and cross-referencing.
- [x] **Hierarchical extraction**: Extend parser to capture full structure (Law -> Chapter -> Section -> Subsection) with chapter-level metadata.
- [x] **Cross-reference parsing**: Detect and store explicit references between sections (e.g., "jf. forbrukerkjøpsloven § 15") as structured metadata.
- [x] **Law-level summaries**: Generate a 2-3 sentence LLM summary for each of the 735 laws, describing scope and target audience. This becomes the routing index.
- [x] **Metadata-rich payloads**: Store `law_id`, `law_title`, `law_short_name`, `chapter_id`, `chapter_title`, `cross_references`, and `url` on every indexed article.
- [x] **Canonical article IDs + source anchors**: Normalize paragraph IDs while preserving original Lovdata anchor IDs.
- [x] **Document typing**: Classify and store `doc_type` (`provision` vs `editorial_note`) for retrieval formatting and ranking policy.

## Phase 6: Multi-Stage Retrieval Pipeline (In Progress)
Build the robust, multi-stage pipeline for full-corpus retrieval.

### Three-Tier Indexing
- [x] **Tier 0 — Law catalog**: Structured table of all 735 laws with summaries for routing. [DONE]
- [ ] **Tier 1 — Chapter-level index**: ~3,000-5,000 vectorized chapter summaries for narrowing within a law.
- [ ] **Tier 2 — Article-level index**: Full article text with hybrid search (dense + sparse) and rich metadata filters. [DONE: Husleieloven indexed]

### Retrieval Stages
- [x] **Query analysis**: Classify query type (section lookup, legal question, concept explanation), extract explicit references, detect legal domain, rewrite follow-ups.
- [x] **Law routing**: Hybrid lexical + reranker law routing with staged uncertainty fallback and explicit fallback-stage diagnostics.
- [x] **Filtered hybrid search**: Search article index filtered to candidate laws, combining semantic and keyword matching.
- [x] **Reranking**: Cross-encoder rescoring of candidates.
- [x] **Editorial context policy**: Keep provisions primary and attach normalized editorial-note payloads per provision (with compatibility fallback).
- [x] **Law coherence filtering**: Remove low-confidence singleton sources from non-dominant laws when score gaps indicate contamination.
- [x] **Law-aware rank fusion**: Deterministic final ordering that combines CE score with routing alignment, affinity, and dominance context.
- [x] **Uncertainty law cap**: Temporary top-law cap for near-tied uncertainty cases to reduce mismatch-heavy fallback spread.
- [ ] **Cross-reference expansion**: Automatically fetch articles referenced by the top results (e.g., "jf. § 9-10").
- [ ] Migrate pipeline orchestration from `create_retrieval_chain` to **LangGraph** for multi-step control flow.

## Phase 7: Validation & Scale
- [ ] Expand eval set to 50-100 questions spanning 10-15 consumer-relevant laws.
- [ ] Test categories: single-law/single-section, single-law/multi-section, cross-law, direct lookups, follow-ups, out-of-scope.
- [ ] Measure each pipeline stage independently (routing accuracy, retrieval recall, reranker precision, answer quality).
- [x] Add law-aware expected source labels (`law_id` + `article_id`) to reduce false-positive citation matches.
- [x] Add post-reindex metadata validation (`missing_doc_type == 0`) and retrieval smoke checks.
- [x] Add retrieval threshold sweep tooling (`scripts/sweep_retrieval_thresholds.py`) for balanced objective tuning.
- [x] Add contamination diagnostics (`scripts/analyze_law_contamination.py`) with routing/coherence insights and hard-cluster summaries.
- [x] Add tiered regression gate baselines (`v1`, `v2`, `v3`) and gate-tier selection in `scripts/check_regression_gates.py`.
- [x] Add calibration diagnostics to threshold sweeps (`calibration_bins`, `expected_calibration_error`).
- [ ] Ensure 0 hallucinated law sections in 50 test queries.
- [ ] Deploy to Streamlit Cloud for private feedback.
- [ ] Index all 735 laws and regulations from Lovdata bulk downloads.

### Current codebase status (Feb 2026)
- Runtime/eval parity has been improved for post-rerank logic (coherence + fusion + uncertainty cap).
- Confidence gating semantics are now explicitly CE-score based.
- Full end-to-end colab validation for all new controls remains an ongoing operational task.

## Future Ideas (Post Full-Corpus)
- **GraphRAG**: Map dependency graph between laws using parsed cross-references.
- **Proactive Alerts**: Notify users when relevant laws change.
- **Document Analysis**: Allow users to upload contracts/agreements for checking against the law.
- **Multi-language**: Support English queries about Norwegian law.
- **Agentic RAG**: Full agent loop where the LLM decides when to retrieve more, ask for clarification, or synthesize across multiple searches.
