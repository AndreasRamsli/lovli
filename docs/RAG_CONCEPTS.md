# RAG & Retrieval Concepts for Lovli

This document explains the core concepts behind the Lovli legal search engine, from basic RAG to the multi-stage retrieval pipeline.

## What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique to give an LLM access to specific, up-to-date data (like Norwegian laws) without retraining the model.

1. **Retrieve**: When a user asks a question, we search our database for the most relevant law sections.
2. **Augment**: We add those sections to the LLM's prompt as context.
3. **Generate**: The LLM answers the question using the provided legal text.

## How Embeddings Work
Embeddings are the "mathematical DNA" of text. They convert words and sentences into long lists of numbers (vectors).

- **Semantic Search**: Unlike keyword search (which looks for exact words), embedding search looks for *meaning*.
- **Example**: A search for "oppsigelse av leilighet" (terminating an apartment lease) will find laws about "leieforholdets opphor" (end of tenancy) because their vectors are mathematically close.

## Why Basic RAG Falls Short for Legal Text

A simple single-pass RAG pipeline (embed question -> find similar documents -> generate answer) has key limitations when applied to a legal corpus of 735+ laws:

- **Similarity is not relevance**: Embedding similarity finds text that *sounds like* the question, not necessarily text that *answers* it. Legal reasoning often requires connecting concepts across sections that don't share surface-level vocabulary.
- **Multi-hop questions**: Questions like "What can a tenant claim for late delivery?" require multiple sections (fulfillment, price reduction, cancellation, damages) that a single query may not surface.
- **Exact references**: If a user asks about "§ 3-5", pure semantic search may not rank that exact section first because the embedding captures meaning, not identifiers.
- **Scale**: With 735 laws, a flat similarity search over all articles returns noisy results. The system needs to narrow down *which law* before searching for *which section*.

## The Lovli Multi-Stage Pipeline

To address these limitations, Lovli uses a multi-stage retrieval pipeline where each stage progressively narrows the search.

### Stage 1: Query Analysis (Implemented)
Before any retrieval, the LLM analyzes the user's question:
- **Query type classification**: Is this a direct section lookup ("What does § 3-5 say?"), a legal question ("Can my landlord evict me?"), or a concept explanation ("What is a deposit?")?
- **Entity extraction**: Detect explicit law/section references, legal domains, and key concepts.
- **Follow-up rewriting**: If this is a follow-up in a conversation, rewrite it to be self-contained (e.g., "What's the deadline for that?" -> "What's the deadline for rent increases under husleieloven?").

### Stage 2: Law Routing (In Progress)
For questions where the relevant law isn't obvious, the system consults a **law catalog** — a compact summary of all 735 Norwegian laws — to identify 1-3 candidate laws. This is a single LLM call against a structured index, not a vector search. [DONE: Catalog generation implemented]

For questions with obvious routing (user mentioned "husleie", or asked about a specific section), this stage is skipped.

### Stage 3: Hybrid Search (Implemented)
Search the article-level index using both:
- **Dense vectors** (BGE-M3): Captures semantic meaning.
- **Sparse vectors** (BM25-style): Captures exact keyword matches, critical for section numbers and legal terms.

Results are filtered by metadata to the candidate laws from Stage 2, reducing noise.

We intentionally **over-retrieve** (k=15) to give the reranker more candidates to work with.

### Stage 4: Reranking (Implemented)
A **cross-encoder reranker** (bge-reranker-v2-m3) rescores all 15 candidates by examining each one alongside the original question. Unlike embedding similarity (which compares vectors independently), a cross-encoder sees the query and document together, producing much more accurate relevance scores.

The top 5 after reranking go to generation.

### Stage 5: Cross-Reference Expansion (Planned)
Norwegian law sections frequently reference other sections (e.g., "jf. § 9-10"). If the top results contain cross-references, the system fetches those referenced articles as additional context. [DONE: Cross-reference parsing implemented]

### Stage 6: Answer Generation (Implemented)
The LLM generates an answer using the retrieved context with:
- **Inline citations**: Every claim references a specific section (e.g., "husleieloven § 3-5").
- **Confidence gating**: If retrieval scores are low, the system says "I couldn't find a clear answer" rather than guessing.
- **Conversation memory**: Chat history is included so the LLM can maintain coherent multi-turn conversations.

## Three-Tier Indexing

The document index is organized in three tiers, each serving a different purpose:

| Tier | Content | Size | Purpose | Storage |
|------|---------|------|---------|---------|
| **Tier 0** | Law catalog (summaries) | ~735 entries | Query routing | Structured data (JSON/DB) |
| **Tier 1** | Chapter summaries | ~3,000-5,000 entries | Narrowing within a law | Qdrant (vectorized) |
| **Tier 2** | Full article text | ~30,000+ entries | Final retrieval | Qdrant (hybrid vectors + metadata) |

## Hybrid Search: Dense + Sparse

BGE-M3 produces both dense and sparse vector representations from the same model:

- **Dense vectors** (1024 dimensions): Capture semantic meaning. Good for questions phrased differently from the law text.
- **Sparse vectors** (bag-of-words style): Capture exact term overlap. Good for specific legal terms, section numbers, and Norwegian domain vocabulary.

Qdrant combines both in a single query, weighting each appropriately. This means a search for "§ 3-5 depositum" benefits from both exact matching on "§ 3-5" (sparse) and semantic understanding of "depositum" (dense).

## Citations & Accuracy
Because we use RAG, every answer can be linked back to a specific Lovdata URL for the source section, ensuring users can verify the information at the original source. The multi-stage pipeline with reranking and confidence gating significantly reduces hallucinated or incorrect citations compared to single-pass retrieval.

## Evaluation Strategy

Each pipeline stage is evaluated independently via LangSmith:
- **Routing accuracy**: Does the system pick the right law(s)?
- **Retrieval recall**: Are the correct articles in the top-k?
- **Reranker precision**: Does reranking improve ordering?
- **Answer quality**: Is the final answer correct, grounded, and well-cited?

This per-stage measurement lets us identify exactly where failures occur and fix the right component.
