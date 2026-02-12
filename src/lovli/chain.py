"""LangChain RAG pipeline for legal question answering."""

import logging
import math
import re
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import CrossEncoder

from .catalog import load_catalog
from .config import Settings, get_settings

logger = logging.getLogger(__name__)
MAX_QUESTION_LENGTH = 1000

# Query rewriting constants
QUERY_REWRITE_MIN_LENGTH_RATIO = 0.5  # Rewritten query must be at least 50% of original length
QUERY_REWRITE_HISTORY_WINDOW = 4  # Last 4 messages (2 turns) for context
_ROUTING_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Confidence gating response (used by both streaming and non-streaming paths)
GATED_RESPONSE = (
    "Jeg fant ikke et klart svar på spørsmålet ditt i lovtekstene. "
    "Kunne du prøve å omformulere spørsmålet eller være mer spesifikk?"
)
NO_RESULTS_RESPONSE = "Beklager, jeg kunne ikke finne informasjon om dette spørsmålet."


def _sigmoid(x: float) -> float:
    """Apply sigmoid function to map raw logits to [0, 1] probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        # exp(-x) overflows for very negative x -> sigmoid approaches 0
        return 0.0


def _tokenize_for_routing(text: str) -> set[str]:
    """Tokenize text for lightweight lexical routing."""
    return set(_ROUTING_TOKEN_RE.findall((text or "").lower()))


class LegalRAGChain:
    """RAG chain for answering legal questions with citations."""

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the RAG chain with Qdrant vector store and LLM.

        Args:
            settings: Application settings (defaults to loading from env)
        """
        self.settings = settings or get_settings()

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name,
        )

        # Initialize Qdrant client
        if self.settings.qdrant_in_memory:
            if self.settings.qdrant_persist_path:
                qdrant_client = QdrantClient(path=self.settings.qdrant_persist_path)
            else:
                qdrant_client = QdrantClient(":memory:")
        else:
            qdrant_client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )

        # Determine if collection uses named vectors (hybrid search with BGE-M3)
        self._uses_named_vectors = False
        if "bge-m3" in self.settings.embedding_model_name.lower():
            try:
                if qdrant_client.collection_exists(self.settings.qdrant_collection_name):
                    collection_info = qdrant_client.get_collection(self.settings.qdrant_collection_name)
                    config = collection_info.config
                    if hasattr(config, 'params') and hasattr(config.params, 'sparse_vectors_config'):
                        sparse_cfg = config.params.sparse_vectors_config
                        if sparse_cfg and len(sparse_cfg) > 0:
                            self._uses_named_vectors = True
            except Exception:
                pass

        # Use hybrid search if collection has sparse vectors configured.
        retrieval_mode = RetrievalMode.DENSE
        if self._uses_named_vectors:
            retrieval_mode = RetrievalMode.HYBRID
            logger.info("Using hybrid search (dense + sparse vectors)")

        # When using named vectors, QdrantVectorStore needs vector_name='dense'
        vs_kwargs = {
            "client": qdrant_client,
            "collection_name": self.settings.qdrant_collection_name,
            "embedding": self.embeddings,
            "retrieval_mode": retrieval_mode,
        }
        if self._uses_named_vectors:
            vs_kwargs["vector_name"] = "dense"
            logger.info("Using named vector 'dense' for QdrantVectorStore (hybrid collection)")
        
        self.vectorstore = QdrantVectorStore(**vs_kwargs)

        # Initialize LLM via OpenRouter
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/lovli",
                "X-Title": "Lovli Legal Assistant",
            },
        )
        
        # Initialize reranker if enabled
        self.reranker = None
        if self.settings.reranker_enabled:
            try:
                logger.info(f"Loading reranker model: {self.settings.reranker_model}")
                self.reranker = CrossEncoder(self.settings.reranker_model)
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker, continuing without reranking: {e}")
                self.reranker = None

        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Du er en hjelpsom assistent som gir KORT og PRESIS informasjon om norsk lov basert på Lovdata.

Regler:
- Svar kortfattet (maks 3-4 setninger for enkle spørsmål)
- Referer til relevante paragrafer (f.eks. § 3-5)
- Ikke gjenta disclaimer i svaret - det vises separat i appen
- Hvis spørsmålet er uklart eller kan gjelde flere områder av loven, still oppfølgingsspørsmål for å avklare hva brukeren faktisk spør om
- Eksempler på uklare spørsmål: "Hva er reglene?", "Hva kan jeg gjøre?", "Er det lovlig?"
- I slike tilfeller, spør konkret: "Hvilket område av husleieloven er du interessert i? For eksempel depositum, oppsigelse, eller husleieøkning?"

Kontekst fra lovtekster:
{context}"""),
            ("human", "{input}"),
        ])

        # Reusable retriever with hybrid search support.
        # Use initial k for over-retrieval (reranker will reduce to final k).
        retrieval_k = self.settings.retrieval_k_initial if self.reranker else self.settings.retrieval_k
        self._retrieval_mode = retrieval_mode
        self._retrieval_k = retrieval_k
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": retrieval_k,
            }
        )

        # Optional Tier-0 law catalog for lightweight routing before retrieval.
        self._law_catalog: list[dict[str, Any]] = []
        if self.settings.law_routing_enabled:
            try:
                self._law_catalog = load_catalog(Path(self.settings.law_catalog_path))
                logger.info(
                    "Loaded law catalog for routing: %s entries from %s",
                    len(self._law_catalog),
                    self.settings.law_catalog_path,
                )
            except Exception as e:
                logger.warning(
                    "Law routing enabled but catalog could not be loaded (%s). "
                    "Falling back to unfiltered retrieval.",
                    e,
                )

    def _validate_question(self, question: str) -> str:
        """Validate and normalize question input."""
        if not question or not question.strip():
            raise ValueError("Vennligst skriv inn et spørsmål.")
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            question = question[:MAX_QUESTION_LENGTH]
        return question

    def _format_context(self, sources: list[Dict[str, Any]]) -> str:
        """Format retrieved sources into a single context string."""
        return "\n\n".join([
            f"Lov: {s['law_title']} (§ {s['article_id']})\n{s.get('content', '')}"
            for s in sources
        ])

    def _extract_sources(
        self, 
        docs: list, 
        include_content: bool = False
    ) -> list[Dict[str, Any]]:
        """Extract deduplicated source metadata from documents."""
        sources = []
        seen_ids = set()
        for doc in docs:
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            article_id = metadata.get("article_id", "Unknown")
            law_id = metadata.get("law_id", "Unknown")
            source_key = (law_id, article_id)
            if source_key not in seen_ids:
                seen_ids.add(source_key)
                source = {
                    "law_id": law_id,
                    "law_title": metadata.get("law_title", "Unknown"),
                    "law_short_name": metadata.get("law_short_name"),
                    "article_id": article_id,
                    "title": metadata.get("title", "Unknown"),
                    "chapter_id": metadata.get("chapter_id"),
                    "chapter_title": metadata.get("chapter_title"),
                    "url": metadata.get("url"),
                }
                if include_content:
                    source["content"] = doc.page_content if hasattr(doc, "page_content") else ""
                sources.append(source)
        return sources

    def _rerank(self, query: str, docs: list, top_k: int | None = None) -> Tuple[list, list[float]]:
        """
        Rerank retrieved documents using cross-encoder reranker.

        Args:
            query: User's question
            docs: List of retrieved documents (LangChain Document objects)
            top_k: Number of top documents to return (defaults to settings.retrieval_k)

        Returns:
            Tuple of (reranked_docs, scores) where scores are sigmoid-normalized
            relevance scores in [0, 1]. Returns empty lists if no docs or reranker.
        """
        if not self.reranker or not docs:
            return docs, []

        top_k = top_k or self.settings.retrieval_k

        # Prepare query-document pairs for reranking
        pairs = [
            [query, doc.page_content if hasattr(doc, "page_content") else str(doc)]
            for doc in docs
        ]

        # Score with cross-encoder
        try:
            raw_scores = self.reranker.predict(pairs)
            # Convert to list if numpy array
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            else:
                raw_scores = list(raw_scores)
        except Exception as e:
            logger.warning(f"Reranking failed, returning original order: {e}")
            return docs, []

        if len(raw_scores) != len(docs):
            logger.warning(
                f"Score count mismatch: {len(raw_scores)} scores for {len(docs)} docs, "
                "returning original order"
            )
            return docs, []

        # Normalize raw logits to [0, 1] via sigmoid.
        # Cross-encoders (including bge-reranker-v2-m3) output unbounded logits,
        # not probabilities. Sigmoid maps them to a consistent [0, 1] range
        # so the confidence threshold in settings is meaningful.
        scores = [_sigmoid(s) for s in raw_scores]

        # Sort by score (descending) and return top_k
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        reranked_docs = [doc for doc, _ in scored_docs[:top_k]]
        reranked_scores = [score for _, score in scored_docs[:top_k]]

        return reranked_docs, reranked_scores

    def _rewrite_query(
        self, 
        question: str, 
        chat_history: list[Dict[str, str]] | None = None,
        max_retries: int = 1
    ) -> str:
        """
        Rewrite a follow-up question using chat history to make it standalone.

        Args:
            question: Current user question
            chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
            max_retries: Maximum number of retry attempts if LLM call fails (default: 1)

        Returns:
            Rewritten standalone question, or original question if rewriting fails
        """
        if not chat_history or len(chat_history) == 0:
            return question
        
        # Build context from last few messages
        context_messages = []
        # Use last N messages (excluding current question which isn't in history yet)
        for msg in chat_history[-QUERY_REWRITE_HISTORY_WINDOW:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:  # Skip empty messages
                context_messages.append(f"{role}: {content}")
        
        context = "\n".join(context_messages)
        
        rewrite_prompt = f"""Gitt denne samtalehistorikken, omskriv det siste spørsmålet som et selvstendig spørsmål som kan forstås uten kontekst.

Samtalehistorikk:
{context}

Siste spørsmål: {question}

Omskriv spørsmålet som et selvstendig spørsmål om norsk lov. Hvis spørsmålet allerede er selvstendig, returner det uendret. Svar kun med det omskrevne spørsmålet, ingen forklaringer."""
        
        # Retry logic for transient failures
        for attempt in range(max_retries + 1):
            try:
                # Use a shorter timeout for query rewriting to avoid blocking
                # Note: ChatOpenAI doesn't support timeout directly, but OpenRouter may
                response = self.llm.invoke(rewrite_prompt)
                rewritten = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                # Validate rewritten query
                if not rewritten or len(rewritten.strip()) == 0:
                    logger.debug("Query rewriting returned empty string, using original")
                    return question
                
                # Fallback to original if rewrite is suspiciously short or identical
                min_length = len(question) * QUERY_REWRITE_MIN_LENGTH_RATIO
                if len(rewritten) < min_length:
                    logger.debug(f"Rewritten query too short ({len(rewritten)} < {min_length}), using original")
                    return question
                
                if rewritten.lower() == question.lower():
                    logger.debug("Rewritten query identical to original, using original")
                    return question
                
                logger.debug(f"Query rewritten: '{question}' -> '{rewritten}'")
                return rewritten
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Query rewriting attempt {attempt + 1} failed, retrying: {e}")
                else:
                    logger.warning(f"Query rewriting failed after {max_retries + 1} attempts, using original query: {e}")
        
        # If all retries failed, return original question
        return question

    def _route_law_ids(self, query: str) -> list[str]:
        """Route query to likely law IDs using simple lexical overlap over catalog metadata."""
        if not self.settings.law_routing_enabled or not self._law_catalog:
            return []

        query_lower = query.lower()
        query_tokens = _tokenize_for_routing(query)
        if not query_tokens:
            return []

        scored: list[tuple[str, int]] = []
        for entry in self._law_catalog:
            law_id = entry.get("law_id")
            if not law_id:
                continue

            title = entry.get("law_title", "")
            short_name = entry.get("law_short_name", "") or ""
            summary = entry.get("summary", "") or ""
            law_ref = entry.get("law_ref", "") or ""
            catalog_text = f"{title} {short_name} {summary} {law_ref}"
            catalog_tokens = _tokenize_for_routing(catalog_text)
            overlap = len(query_tokens & catalog_tokens)

            # Bonus for direct short-name mention, e.g. "husleieloven".
            if short_name and short_name.lower() in query_lower:
                overlap += 3

            if overlap >= self.settings.law_routing_min_token_overlap:
                scored.append((law_id, overlap))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        max_candidates = self.settings.law_routing_max_candidates
        routed = [law_id for law_id, _ in scored[:max_candidates]]
        logger.debug("Law routing candidates for '%s': %s", query, routed)
        return routed

    def _build_law_filter(self, law_ids: list[str]) -> qdrant_models.Filter | None:
        """Build Qdrant payload filter for metadata.law_id."""
        if not law_ids:
            return None
        return qdrant_models.Filter(
            should=[
                qdrant_models.FieldCondition(
                    key="metadata.law_id",
                    match=qdrant_models.MatchValue(value=law_id),
                )
                for law_id in law_ids
            ]
        )

    def _invoke_retriever(self, query: str, routed_law_ids: list[str] | None = None):
        """Invoke retriever with optional law_id filter; fallback to unfiltered on errors."""
        if not routed_law_ids:
            return self.retriever.invoke(query)

        law_filter = self._build_law_filter(routed_law_ids)
        try:
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self._retrieval_k,
                    "filter": law_filter,
                }
            )
            docs = filtered_retriever.invoke(query)
            if docs:
                return docs
        except Exception as e:
            logger.warning("Filtered retrieval failed (%s); using unfiltered retrieval.", e)

        return self.retriever.invoke(query)

    def _apply_reranker_doc_filter(
        self, docs: list, scores: list[float]
    ) -> tuple[list, list[float]]:
        """Filter reranked docs by per-document score with a floor on minimum sources."""
        if not docs or not scores:
            return docs, scores

        min_doc_score = self.settings.reranker_min_doc_score
        min_sources = min(self.settings.reranker_min_sources, len(docs))

        kept = [(doc, score) for doc, score in zip(docs, scores) if score >= min_doc_score]
        if len(kept) < min_sources:
            kept = list(zip(docs, scores))[:min_sources]

        filtered_docs = [doc for doc, _ in kept]
        filtered_scores = [score for _, score in kept]
        dropped = len(docs) - len(filtered_docs)
        if dropped > 0:
            logger.debug(
                "Dropped %s low-score reranked docs (min_doc_score=%.2f)",
                dropped,
                min_doc_score,
            )
        return filtered_docs, filtered_scores

    def retrieve(
        self, question: str, chat_history: list[Dict[str, str]] | None = None
    ) -> Tuple[list[Dict[str, Any]], float | None, list[float]]:
        """
        Retrieve relevant legal documents for a question.

        Args:
            question: User's legal question
            chat_history: Optional chat history for query rewriting

        Returns:
            Tuple of (sources, top_score, reranker_scores) where top_score is the highest
            reranker score (None if no reranking).

        Raises:
            ValueError: If question is empty
        """
        question = self._validate_question(question)
        
        # Rewrite query if chat history is provided
        retrieval_query = self._rewrite_query(question, chat_history) if chat_history else question
        
        # Optional Tier-0 law routing before retrieval.
        routed_law_ids = self._route_law_ids(retrieval_query)
        docs = self._invoke_retriever(retrieval_query, routed_law_ids=routed_law_ids)
        
        # Deduplicate documents by article_id before reranking to avoid wasted compute
        # and ensure we get the full requested count after reranking
        if docs:
            original_count = len(docs)
            seen_article_keys = set()
            deduplicated_docs = []
            for doc in docs:
                metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                article_id = metadata.get("article_id")
                law_id = metadata.get("law_id")
                article_key = (law_id, article_id)
                if article_id and article_key not in seen_article_keys:
                    seen_article_keys.add(article_key)
                    deduplicated_docs.append(doc)
                elif not article_id:
                    # If article_id is missing, include the doc anyway (shouldn't happen)
                    deduplicated_docs.append(doc)
            docs = deduplicated_docs
            if len(deduplicated_docs) < original_count:
                logger.debug(f"Deduplicated {original_count} docs to {len(deduplicated_docs)} before reranking")
        
        top_score = None
        scores: list[float] = []
        # Rerank if enabled
        if self.reranker and docs:
            docs, scores = self._rerank(retrieval_query, docs, top_k=self.settings.retrieval_k)
            docs, scores = self._apply_reranker_doc_filter(docs, scores)
            top_score = scores[0] if scores else None
            if top_score is not None:
                logger.debug(f"Reranked {len(docs)} documents, top score: {top_score:.3f}")
            else:
                logger.debug(f"Reranking returned no scores for {len(docs)} documents")
        
        sources = self._extract_sources(docs, include_content=True)
        return sources, top_score, scores
    
    def should_gate_answer(self, top_score: float | None, scores: list[float] | None = None) -> bool:
        """
        Check if answer should be gated due to low confidence.

        Args:
            top_score: Highest reranker score (None if reranking disabled)
            scores: Optional full reranker score list for ambiguity gating

        Returns:
            True if answer should be gated (low confidence)
        """
        if top_score is None:
            # No reranking, don't gate
            return False
        if top_score < self.settings.reranker_confidence_threshold:
            return True
        if (
            self.settings.reranker_ambiguity_gating_enabled
            and scores
            and len(scores) >= 2
            and scores[0] <= self.settings.reranker_ambiguity_top_score_ceiling
        ):
            score_gap = scores[0] - scores[1]
            if score_gap < self.settings.reranker_ambiguity_min_gap:
                return True
        return False

    def stream_answer(
        self,
        question: str,
        sources: list[Dict[str, Any]],
        top_score: float | None = None,
        scores: list[float] | None = None,
    ) -> Iterator[str]:
        """
        Stream an answer for a question given retrieved sources.

        Args:
            question: User's legal question
            sources: List of retrieved sources from retrieve()
            top_score: Highest reranker score for confidence gating
            scores: Full reranker score list for ambiguity gating

        Yields:
            Tokens of the LLM response
        """
        # Confidence gating: if reranker score is too low, return canned response
        if self.should_gate_answer(top_score, scores=scores):
            yield GATED_RESPONSE
            return

        if not sources:
            yield NO_RESULTS_RESPONSE
            return
        
        context = self._format_context(sources)

        # Use format_messages to preserve system/human role separation
        messages = self.prompt_template.format_messages(context=context, input=question)

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def query(self, question: str, chat_history: list[Dict[str, str]] | None = None) -> Dict[str, Any]:
        """
        Query the RAG chain with a legal question (non-streaming).
        
        Uses the same pipeline as retrieve() + stream_answer() for consistency.

        Args:
            question: User's legal question in Norwegian
            chat_history: Optional chat history for query rewriting

        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        try:
            # Use the same retrieval pipeline as streaming API
            # retrieve() handles validation internally
            sources, top_score, scores = self.retrieve(question, chat_history=chat_history)
        except ValueError as e:
            # Handle validation errors from retrieve()
            return {"answer": str(e), "sources": []}
        
        # Check confidence gating
        if self.should_gate_answer(top_score, scores=scores):
            return {"answer": GATED_RESPONSE, "sources": []}

        if not sources:
            return {"answer": NO_RESULTS_RESPONSE, "sources": []}
        
        # Generate answer using same prompt template as streaming
        context = self._format_context(sources)
        
        messages = self.prompt_template.format_messages(context=context, input=question)
        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # Return sources without content for consistency
        sources_for_return = [
            {k: v for k, v in s.items() if k != "content"}
            for s in sources
        ]
        
        return {"answer": answer, "sources": sources_for_return}
