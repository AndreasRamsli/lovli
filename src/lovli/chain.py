"""LangChain RAG pipeline for legal question answering."""

import logging
from pathlib import Path
from typing import Dict, Any, Iterator, Tuple
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import CrossEncoder

from .catalog import load_catalog
from .config import Settings, get_settings
from .editorial import (
    collect_provision_article_pairs,
    collect_provision_law_chapter_pairs,
    dedupe_by_law_article,
)
from .scoring import (
    apply_uncertainty_law_cap,
    build_law_aware_rank_fusion,
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    normalize_law_ref,
    _infer_doc_type,
)
from .routing import (
    build_law_embedding_index,
    build_routing_entries,
    score_all_laws_embedding,
    score_law_candidates_lexical,
    score_law_candidates_reranker,
    compute_routing_alignment,
    tokenize_for_routing as _tokenize_for_routing,
    normalize_keyword_term as _normalize_keyword_term,
    normalize_law_mention as _normalize_law_mention,
)
from .reranking import (
    build_reranker_document_text as _build_reranker_document_text_fn,
    rerank_documents,
    filter_reranked_docs,
    should_gate_answer as _should_gate_answer_fn,
)

logger = logging.getLogger(__name__)
MAX_QUESTION_LENGTH = 1000

# Query rewriting constants
QUERY_REWRITE_MIN_LENGTH_RATIO = 0.5  # Rewritten query must be at least 50% of original length
QUERY_REWRITE_HISTORY_WINDOW = 4  # Last 4 messages (2 turns) for context

# Confidence gating response (used by both streaming and non-streaming paths)
GATED_RESPONSE = (
    "Jeg fant ikke et klart svar på spørsmålet ditt i lovtekstene. "
    "Kunne du prøve å omformulere spørsmålet eller være mer spesifikk?"
)
NO_RESULTS_RESPONSE = "Beklager, jeg kunne ikke finne informasjon om dette spørsmålet."


def _editorial_note_payload(doc: Document) -> Dict[str, Any]:
    """Normalize an editorial note document into a compact metadata payload."""
    metadata = doc.metadata if hasattr(doc, "metadata") else {}
    content = (doc.page_content if hasattr(doc, "page_content") else "") or ""
    return {
        "article_id": metadata.get("article_id", ""),
        "content": content.strip(),
        "title": metadata.get("title", ""),
        "source_anchor_id": metadata.get("source_anchor_id"),
        "url": metadata.get("url"),
        "chapter_id": metadata.get("chapter_id"),
    }


def _normalize_editorial_notes(
    notes: list[Dict[str, Any]],
    max_notes: int,
    max_chars: int,
) -> list[Dict[str, Any]]:
    """Apply editorial note payload contract, dedupe, and deterministic ordering."""
    normalized: list[Dict[str, Any]] = []
    for note in notes or []:
        article_id = (note.get("article_id") or "").strip()
        content = (note.get("content") or "").strip()
        if max_chars > 0:
            content = content[:max_chars]
        if not article_id or not content:
            continue
        normalized.append(
            {
                "article_id": article_id,
                "content": content,
                "title": note.get("title"),
                "source_anchor_id": note.get("source_anchor_id"),
                "url": note.get("url"),
                "chapter_id": note.get("chapter_id"),
            }
        )

    normalized.sort(
        key=lambda item: (
            (item.get("source_anchor_id") or ""),
            (item.get("article_id") or ""),
            (item.get("content") or ""),
        )
    )
    deduped: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for note in normalized:
        key = ((note.get("article_id") or "").strip(), (note.get("content") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(note)
    return deduped[:max_notes] if max_notes > 0 else deduped


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
                    collection_info = qdrant_client.get_collection(
                        self.settings.qdrant_collection_name
                    )
                    config = collection_info.config
                    if hasattr(config, "params") and hasattr(
                        config.params, "sparse_vectors_config"
                    ):
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
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(
                    "Loading reranker model: %s on %s", self.settings.reranker_model, device
                )
                # Force fp32 via model_kwargs (automodel_args was renamed in sentence-transformers>=5)
                # This prevents fp16 sigmoid collapse on GPU where logits compress near zero.
                self.reranker = CrossEncoder(
                    self.settings.reranker_model,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float32},
                )
                logger.info("Reranker loaded successfully on %s", device)
            except Exception as e:
                logger.warning("Failed to load reranker, continuing without reranking: %s", e)
                self.reranker = None

        # Prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Du er en hjelpsom assistent som gir KORT og PRESIS informasjon om norsk lov basert på Lovdata.

Regler:
- Svar kortfattet (maks 3-4 setninger for enkle spørsmål)
- Referer til relevante paragrafer (f.eks. § 3-5)
- Ikke gjenta disclaimer i svaret - det vises separat i appen
- Hvis spørsmålet er uklart eller kan gjelde flere områder av loven, still oppfølgingsspørsmål for å avklare hva brukeren faktisk spør om
- Eksempler på uklare spørsmål: "Hva er reglene?", "Hva kan jeg gjøre?", "Er det lovlig?"
- I slike tilfeller, spør konkret: "Hvilket område av husleieloven er du interessert i? For eksempel depositum, oppsigelse, eller husleieøkning?"

Kontekst fra lovtekster:
{context}""",
                ),
                ("human", "{input}"),
            ]
        )

        # Reusable retriever with hybrid search support.
        # Use initial k for over-retrieval (reranker will reduce to final k).
        retrieval_k = (
            self.settings.retrieval_k_initial if self.reranker else self.settings.retrieval_k
        )
        self._retrieval_mode = retrieval_mode
        self._retrieval_k = retrieval_k
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": retrieval_k,
            }
        )

        # Optional Tier-0 law catalog for lightweight routing before retrieval.
        self._law_catalog: list[dict[str, Any]] = []
        self._law_catalog_entries: list[dict[str, Any]] = []
        self._law_ref_to_id: dict[str, str] = {}
        self._last_routing_diagnostics: dict[str, Any] = {}
        self._last_coherence_diagnostics: dict[str, Any] = {}
        self._last_rank_fusion_diagnostics: dict[str, Any] = {}
        if self.settings.law_routing_enabled:
            try:
                self._law_catalog = load_catalog(Path(self.settings.law_catalog_path))
                self._law_catalog_entries = self._build_routing_entries(self._law_catalog)
                self._law_ref_to_id = {
                    normalize_law_ref(item.get("law_ref") or ""): (item.get("law_id") or "").strip()
                    for item in self._law_catalog
                    if normalize_law_ref(item.get("law_ref") or "")
                    and (item.get("law_id") or "").strip()
                }
                logger.info(
                    "Loaded law catalog for routing: %s entries from %s",
                    len(self._law_catalog),
                    self.settings.law_catalog_path,
                )
                # Build embedding index for semantic law routing if enabled.
                # Law texts are embedded once here and cosine similarity is computed
                # per query — much faster than re-embedding at query time.
                #
                # TODO(perf): On CPU this takes ~30-90s for 4427 laws (BGE-M3 is
                # 570M params). For production CPU deployments, serialize the index
                # to disk (numpy + catalog checksum) and reload on subsequent starts.
                # On GPU (H100/A100) the full embed takes ~2-4s — acceptable.
                # Track at: https://github.com/AndreasRamsli/lovli/issues
                if self.settings.law_routing_embedding_enabled:
                    try:
                        self._law_catalog_entries = build_law_embedding_index(
                            self._law_catalog_entries,
                            embed_documents=self.embeddings.embed_documents,
                            text_field=self.settings.law_routing_embedding_text_field,
                        )
                    except Exception as emb_err:
                        logger.warning(
                            "Law embedding index build failed (%s). "
                            "Routing will fall back to lexical-only.",
                            emb_err,
                        )
            except Exception as e:
                logger.warning(
                    "Law routing enabled but catalog could not be loaded (%s). "
                    "Falling back to unfiltered retrieval.",
                    e,
                )

    def _build_routing_entries(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Precompute routing text/tokens for fast hybrid law routing."""
        return build_routing_entries(catalog)

    def _score_law_candidates_lexical(self, query: str) -> list[dict[str, Any]]:
        """Score catalog entries lexically before optional reranker-based law scoring."""
        return score_law_candidates_lexical(
            query,
            self._law_catalog_entries,
            min_token_overlap=self.settings.law_routing_min_token_overlap,
            prefilter_k=self.settings.law_routing_prefilter_k,
        )

    def _score_law_candidates_reranker(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Score law candidates for routing — embedding hybrid, cross-encoder, or lexical.

        Priority order:
        1. Embedding hybrid (default, ``law_routing_embedding_enabled=True``):
           BGE-M3 cosine similarity blended with lexical overlap. Meaningful
           semantic scores; avoids the cross-encoder task mismatch for catalog texts.
        2. Cross-encoder reranker (``law_routing_reranker_enabled=True``):
           bge-reranker-v2-m3. Only useful if a routing-specific model is loaded;
           the document reranker produces near-zero logits for catalog summary pairs.
        3. Lexical only (both disabled): all ``law_reranker_score`` set to None,
           routing falls through to token-overlap top-k selection.
        """
        # --- Embedding hybrid path ---
        # NOTE: when law_routing_embedding_enabled is True, _route_law_ids bypasses
        # this method entirely and calls score_all_laws_embedding() directly (ANN pass
        # over all laws, no lexical prefilter). This block is only reached when embedding
        # is enabled but _route_law_ids fell back to the lexical path due to an error.

        # --- Cross-encoder path (opt-in, rarely useful for catalog texts) ---
        if self.settings.law_routing_reranker_enabled:
            return score_law_candidates_reranker(
                query,
                candidates,
                self.reranker,
                dualpass_enabled=bool(self.settings.law_routing_summary_dualpass_enabled),
                summary_weight=max(0.0, float(self.settings.law_routing_dualpass_summary_weight)),
                title_weight=max(0.0, float(self.settings.law_routing_dualpass_title_weight)),
                fulltext_weight=max(0.0, float(self.settings.law_routing_dualpass_fulltext_weight)),
            )

        # --- Lexical-only fallback ---
        for candidate in candidates:
            candidate["law_reranker_score"] = None
        return candidates

    def _validate_question(self, question: str) -> str:
        """Validate and normalize question input."""
        if not question or not question.strip():
            raise ValueError("Vennligst skriv inn et spørsmål.")
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            question = question[:MAX_QUESTION_LENGTH]
        return question

    def _format_context(self, sources: list[Dict[str, Any]]) -> str:
        """Format retrieved sources with editorial notes attached to each provision."""
        if not sources:
            return ""

        provision_blocks: list[str] = []
        for source in sources:
            if source.get("doc_type") == "editorial_note":
                continue

            block = (
                f"Lov: {source.get('law_title', 'Unknown')} (§ {source.get('article_id', 'Unknown')})\n"
                f"{source.get('content', '')}"
            )
            editorial_notes = _normalize_editorial_notes(
                source.get("editorial_notes") or [],
                max_notes=self.settings.editorial_notes_per_provision_cap,
                max_chars=self.settings.editorial_note_max_chars,
            )
            if editorial_notes:
                history_lines = []
                for note in editorial_notes:
                    note_id = note.get("article_id", "")
                    note_content = (note.get("content") or "").strip()
                    if note_id and note_content:
                        history_lines.append(f"- [{note_id}] {note_content}")
                    elif note_content:
                        history_lines.append(f"- {note_content}")
                if history_lines:
                    block += "\n\nEndringshistorikk:\n" + "\n".join(history_lines)
            provision_blocks.append(block)

        if not provision_blocks:
            return ""
        return "Lovgrunnlag:\n" + "\n\n".join(provision_blocks)

    def _extract_sources(self, docs: list, include_content: bool = False) -> list[Dict[str, Any]]:
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
                    "source_anchor_id": metadata.get("source_anchor_id"),
                    "doc_type": _infer_doc_type(metadata),
                    "editorial_notes": _normalize_editorial_notes(
                        metadata.get("editorial_notes", []) or [],
                        max_notes=self.settings.editorial_notes_per_provision_cap,
                        max_chars=self.settings.editorial_note_max_chars,
                    ),
                }
                if include_content:
                    source["content"] = doc.page_content if hasattr(doc, "page_content") else ""
                sources.append(source)
        return sources

    def _fetch_editorial_for_provisions(
        self,
        provision_pairs: list[tuple[str, str]],
        per_provision_cap: int = 3,
    ) -> list:
        """Fetch editorial notes linked directly to retrieved provisions."""
        if not provision_pairs:
            return []

        client = getattr(self.vectorstore, "client", None)
        if client is None:
            return []

        editorial_docs: list = []
        for law_id, article_id in provision_pairs:
            try:
                offset = None
                provision_docs: list = []
                while True:
                    points, offset = client.scroll(
                        collection_name=self.settings.qdrant_collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="metadata.law_id",
                                    match=qdrant_models.MatchValue(value=law_id),
                                ),
                                qdrant_models.FieldCondition(
                                    key="metadata.linked_provision_id",
                                    match=qdrant_models.MatchValue(value=article_id),
                                ),
                                qdrant_models.FieldCondition(
                                    key="metadata.doc_type",
                                    match=qdrant_models.MatchValue(value="editorial_note"),
                                ),
                            ]
                        ),
                        limit=128,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in points:
                        payload = getattr(point, "payload", {}) or {}
                        metadata = payload.get("metadata", {}) or {}
                        if _infer_doc_type(metadata) != "editorial_note":
                            continue
                        provision_docs.append(
                            Document(
                                page_content=payload.get("page_content", ""),
                                metadata=metadata,
                            )
                        )
                    if offset is None:
                        break
                provision_docs.sort(
                    key=lambda d: (
                        d.metadata.get("source_anchor_id", "") or d.metadata.get("article_id", ""),
                        d.metadata.get("article_id", ""),
                    )
                )
                editorial_docs.extend(provision_docs[:per_provision_cap])
            except Exception as exc:
                logger.warning(
                    "Skipping linked editorial fetch (missing payload indexes or filter unsupported): %s",
                    exc,
                )
                return []
        return dedupe_by_law_article(editorial_docs)

    def _fetch_editorial_for_chapters(
        self, law_chapter_pairs: list[tuple[str, str]], per_chapter_cap: int = 2
    ) -> list:
        """
        Fetch editorial notes for retrieved provision chapters.

        Falls back to empty results when filtered payload search is unavailable
        (for example missing payload indexes on cloud collections).
        """
        if not law_chapter_pairs:
            return []

        client = getattr(self.vectorstore, "client", None)
        if client is None:
            return []

        editorial_docs: list = []
        for law_id, chapter_id in law_chapter_pairs:
            try:
                offset = None
                chapter_docs: list = []
                while True:
                    points, offset = client.scroll(
                        collection_name=self.settings.qdrant_collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="metadata.law_id",
                                    match=qdrant_models.MatchValue(value=law_id),
                                ),
                                qdrant_models.FieldCondition(
                                    key="metadata.chapter_id",
                                    match=qdrant_models.MatchValue(value=chapter_id),
                                ),
                                qdrant_models.FieldCondition(
                                    key="metadata.doc_type",
                                    match=qdrant_models.MatchValue(value="editorial_note"),
                                ),
                            ]
                        ),
                        limit=128,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in points:
                        payload = getattr(point, "payload", {}) or {}
                        metadata = payload.get("metadata", {}) or {}
                        if _infer_doc_type(metadata) != "editorial_note":
                            continue
                        chapter_docs.append(
                            Document(
                                page_content=payload.get("page_content", ""),
                                metadata=metadata,
                            )
                        )
                    if offset is None:
                        break
                # Deterministic ordering and chapter-level cap to avoid flooding.
                chapter_docs.sort(
                    key=lambda d: (
                        d.metadata.get("source_anchor_id", "") or d.metadata.get("article_id", ""),
                        d.metadata.get("article_id", ""),
                    )
                )
                editorial_docs.extend(chapter_docs[:per_chapter_cap])
            except Exception as exc:
                logger.warning(
                    "Skipping editorial injection (missing payload indexes or filter unsupported): %s",
                    exc,
                )
                return []

        return dedupe_by_law_article(editorial_docs)

    def _attach_editorial_to_provisions(
        self, docs: list, scores: list[float] | None = None
    ) -> tuple[list, list[float]]:
        """Keep provisions only and normalize attached editorial payloads."""
        if not docs:
            return docs, scores or []

        has_aligned_scores = bool(scores) and len(scores) == len(docs)
        provisions: list = []
        provision_scores: list[float] = []
        if has_aligned_scores:
            for doc, score in zip(docs, scores):
                metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                if _infer_doc_type(metadata) != "editorial_note":
                    provisions.append(doc)
                    provision_scores.append(score)
        else:
            for doc in docs:
                metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                if _infer_doc_type(metadata) != "editorial_note":
                    provisions.append(doc)
        if not provisions:
            return provisions, provision_scores if has_aligned_scores else []

        has_inline_payload = False
        for provision_doc in provisions:
            p_meta = provision_doc.metadata if hasattr(provision_doc, "metadata") else {}
            normalized = _normalize_editorial_notes(
                p_meta.get("editorial_notes", []) or [],
                max_notes=self.settings.editorial_notes_per_provision_cap,
                max_chars=self.settings.editorial_note_max_chars,
            )
            if normalized:
                has_inline_payload = True
            p_meta["editorial_notes"] = normalized

        if has_inline_payload or not self.settings.editorial_v2_compat_mode:
            return provisions, provision_scores if has_aligned_scores else []

        editorial_by_provision: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
        editorial_by_chapter: Dict[tuple[str, str], list[Dict[str, Any]]] = {}

        provision_pairs = collect_provision_article_pairs(provisions)
        editorial_docs = self._fetch_editorial_for_provisions(provision_pairs)
        if not editorial_docs:
            law_chapter_pairs = collect_provision_law_chapter_pairs(provisions)
            editorial_docs = self._fetch_editorial_for_chapters(
                law_chapter_pairs, per_chapter_cap=2
            )

        for editorial_doc in editorial_docs:
            e_meta = editorial_doc.metadata if hasattr(editorial_doc, "metadata") else {}
            note_payload = _editorial_note_payload(editorial_doc)
            law_id = (e_meta.get("law_id") or "").strip()
            linked_provision_id = (e_meta.get("linked_provision_id") or "").strip()
            chapter_id = (e_meta.get("chapter_id") or "").strip()
            if law_id and linked_provision_id:
                editorial_by_provision.setdefault((law_id, linked_provision_id), []).append(
                    note_payload
                )
            elif law_id and chapter_id:
                editorial_by_chapter.setdefault((law_id, chapter_id), []).append(note_payload)

        chapter_to_provision_keys: Dict[tuple[str, str], list[tuple[str, str]]] = {}
        for provision_doc in provisions:
            p_meta = provision_doc.metadata if hasattr(provision_doc, "metadata") else {}
            law_id = (p_meta.get("law_id") or "").strip()
            article_id = (p_meta.get("article_id") or "").strip()
            chapter_id = (p_meta.get("chapter_id") or "").strip()
            if law_id and chapter_id and article_id:
                chapter_to_provision_keys.setdefault((law_id, chapter_id), []).append(
                    (law_id, article_id)
                )

        for provision_doc in provisions:
            p_meta = provision_doc.metadata if hasattr(provision_doc, "metadata") else {}
            law_id = (p_meta.get("law_id") or "").strip()
            article_id = (p_meta.get("article_id") or "").strip()
            chapter_id = (p_meta.get("chapter_id") or "").strip()
            notes = list(editorial_by_provision.get((law_id, article_id), []))
            # Chapter fallback is only safe when exactly one retrieved provision
            # exists in that chapter; otherwise note ownership is ambiguous.
            chapter_key = (law_id, chapter_id)
            if (
                not notes
                and law_id
                and chapter_id
                and len(chapter_to_provision_keys.get(chapter_key, [])) == 1
            ):
                notes = list(editorial_by_chapter.get(chapter_key, []))
            p_meta["editorial_notes"] = _normalize_editorial_notes(
                notes,
                max_notes=self.settings.editorial_notes_per_provision_cap,
                max_chars=self.settings.editorial_note_max_chars,
            )
        return provisions, provision_scores if has_aligned_scores else []

    def _prioritize_doc_types(
        self, docs: list, scores: list[float], retrieval_query: str
    ) -> tuple[list, list[float]]:
        """Keep provisions only; editorial context is attached within provision metadata."""
        if not docs:
            return docs, scores
        if scores and len(scores) == len(docs):
            return docs, scores
        return docs, []

    def _build_reranker_document_text(self, doc: Any) -> str:
        """Build reranker text with optional metadata prefix for better disambiguation."""
        return _build_reranker_document_text_fn(
            doc,
            context_enrichment_enabled=self.settings.reranker_context_enrichment_enabled,
            context_max_prefix_chars=self.settings.reranker_context_max_prefix_chars,
        )

    def _rerank(self, query: str, docs: list, top_k: int | None = None) -> Tuple[list, list[float]]:
        """
        Rerank retrieved documents using cross-encoder reranker.

        Returns:
            Tuple of (reranked_docs, scores) where scores are sigmoid-normalized
            relevance scores in [0, 1]. Returns empty lists if no docs or reranker.
        """
        if not self.reranker or not docs:
            return docs, []
        effective_k = top_k or self.settings.retrieval_k
        return rerank_documents(
            query,
            docs,
            self.reranker,
            effective_k,
            context_enrichment_enabled=self.settings.reranker_context_enrichment_enabled,
            context_max_prefix_chars=self.settings.reranker_context_max_prefix_chars,
        )

    def _rewrite_query(
        self, question: str, chat_history: list[Dict[str, str]] | None = None, max_retries: int = 1
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
                rewritten = (
                    response.content.strip()
                    if hasattr(response, "content")
                    else str(response).strip()
                )

                # Validate rewritten query
                if not rewritten or len(rewritten.strip()) == 0:
                    logger.debug("Query rewriting returned empty string, using original")
                    return question

                # Fallback to original if rewrite is suspiciously short or identical
                min_length = len(question) * QUERY_REWRITE_MIN_LENGTH_RATIO
                if len(rewritten) < min_length:
                    logger.debug(
                        f"Rewritten query too short ({len(rewritten)} < {min_length}), using original"
                    )
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
                    logger.warning(
                        f"Query rewriting failed after {max_retries + 1} attempts, using original query: {e}"
                    )

        # If all retries failed, return original question
        return question

    def _route_law_ids(self, query: str) -> list[str]:
        """Route query to likely laws using hybrid lexical + law-level reranker scoring."""
        if not self.settings.law_routing_enabled or not self._law_catalog_entries:
            self._last_routing_diagnostics = {
                "enabled": bool(self.settings.law_routing_enabled),
                "reason": "routing_disabled_or_empty_catalog",
                "routed_law_ids": [],
            }
            return []
        lexical_candidates: list[dict[str, Any]] = []
        # When embedding routing is enabled, bypass the lexical token-overlap
        # prefilter and score ALL catalog entries directly by cosine similarity.
        # The lexical prefilter drops laws whose catalog text doesn't share exact
        # tokens with the query (e.g. husleieloven is absent for "depositum" or
        # "husleien" queries because the catalog uses uninflected forms).
        # Scoring all 4427 laws is fast on GPU (~2-4ms for a single matmul).
        if self.settings.law_routing_embedding_enabled and self._law_catalog_entries:
            has_embeddings = any(
                c.get("embedding") is not None for c in self._law_catalog_entries[:5]
            )
            if has_embeddings:
                try:
                    query_embedding = self.embeddings.embed_query(query)
                    query_norm = _normalize_law_mention(query)
                    scored_candidates = score_all_laws_embedding(
                        query_embedding,
                        self._law_catalog_entries,
                        top_k=self.settings.law_routing_prefilter_k,
                        query_norm=query_norm,
                    )
                    if not scored_candidates:
                        self._last_routing_diagnostics = {
                            "enabled": True,
                            "reason": "no_embedding_candidates",
                            "routed_law_ids": [],
                        }
                        return []
                except Exception as exc:
                    logger.warning(
                        "Embedding law routing failed (%s); falling back to lexical.", exc
                    )
                    scored_candidates = None
            else:
                logger.warning(
                    "Embedding routing enabled but no embeddings found in catalog entries. "
                    "Was build_law_embedding_index called? Falling back to lexical."
                )
                scored_candidates = None

            if scored_candidates is not None:
                # Skip the rest of the lexical path
                pass
            else:
                lexical_candidates = self._score_law_candidates_lexical(query)
                if not lexical_candidates:
                    self._last_routing_diagnostics = {
                        "enabled": True,
                        "reason": "no_lexical_candidates",
                        "routed_law_ids": [],
                    }
                    return []
                scored_candidates = self._score_law_candidates_reranker(query, lexical_candidates)
        else:
            lexical_candidates = self._score_law_candidates_lexical(query)
            if not lexical_candidates:
                self._last_routing_diagnostics = {
                    "enabled": True,
                    "reason": "no_lexical_candidates",
                    "routed_law_ids": [],
                }
                return []
            scored_candidates = self._score_law_candidates_reranker(query, lexical_candidates)

        min_confidence = self.settings.law_routing_min_confidence
        rerank_top_k = self.settings.law_routing_rerank_top_k
        score_window = float(getattr(self.settings, "law_routing_score_window", 0.15))
        lexical_top_k = self.settings.law_routing_max_candidates
        fallback_max_laws = self.settings.law_routing_fallback_max_laws
        fallback_min_lexical = self.settings.law_routing_fallback_min_lexical_support

        top_score = None
        second_score = None
        score_gap = None
        score_mode = "lexical_only"
        if scored_candidates:
            top_score = scored_candidates[0].get("law_reranker_score")
            second_score = (
                scored_candidates[1].get("law_reranker_score")
                if len(scored_candidates) > 1
                else None
            )
            if top_score is not None:
                if second_score is not None:
                    score_gap = top_score - second_score
                # Determine score_mode independently of whether second_score exists
                if self.settings.law_routing_embedding_enabled and any(
                    c.get("embedding") is not None for c in scored_candidates[:5]
                ):
                    score_mode = "embedding_hybrid"
                elif self.settings.law_routing_summary_dualpass_enabled:
                    score_mode = "reranker_dualpass"
                else:
                    score_mode = "reranker"

        def _is_uncertain(candidates: list[dict[str, Any]]) -> bool:
            if not candidates:
                return False
            # Uncertainty detection applies to any scored path (embedding or reranker),
            # not just the cross-encoder. Previously gated on self.reranker which silently
            # disabled uncertainty handling for the embedding hybrid path.
            has_scores = any(c.get("law_reranker_score") is not None for c in candidates)
            if not has_scores:
                return False
            top = candidates[0]
            top_score = top.get("law_reranker_score")
            if top_score is None:
                return False
            if len(candidates) < 2:
                return bool(top_score <= self.settings.law_routing_uncertainty_top_score_ceiling)
            second_score = candidates[1].get("law_reranker_score")
            if second_score is None:
                return bool(top_score <= self.settings.law_routing_uncertainty_top_score_ceiling)
            gap = top_score - second_score
            return (
                top_score <= self.settings.law_routing_uncertainty_top_score_ceiling
                and gap < self.settings.law_routing_uncertainty_min_gap
            )

        fallback_mode = None
        uncertainty_selection: dict[str, Any] | None = None
        if self.reranker:
            selected = [
                candidate
                for candidate in scored_candidates
                if candidate.get("direct_mention")
                or (
                    candidate.get("law_reranker_score") is not None
                    and candidate.get("law_reranker_score", 0.0) >= min_confidence
                )
            ]
            if not selected:
                selected = scored_candidates[: max(lexical_top_k, rerank_top_k)]
            if _is_uncertain(scored_candidates):
                broadened: list[dict[str, Any]] = []
                excluded = 0
                relaxed_rerank_floor = max(0.0, float(min_confidence) * 0.80)
                # In the ANN path all candidates have lexical_score=0 (no lexical
                # gate was applied), so gating on lexical_score would reject every
                # candidate. Use rerank_score alone when no lexical signal exists.
                ann_path = self.settings.law_routing_embedding_enabled and all(
                    int(c.get("lexical_score", 0)) == 0 for c in scored_candidates[:5]
                )
                for candidate in scored_candidates:
                    rerank_score = candidate.get("law_reranker_score")
                    rerank_ok = (
                        rerank_score is not None and float(rerank_score) >= relaxed_rerank_floor
                    )
                    direct_mention = bool(candidate.get("direct_mention"))
                    if ann_path:
                        # ANN path: admit by score or direct mention alone
                        if direct_mention or rerank_ok:
                            broadened.append(candidate)
                        else:
                            excluded += 1
                    else:
                        lexical_ok = int(candidate.get("lexical_score", 0)) >= fallback_min_lexical
                        if lexical_ok and (direct_mention or rerank_ok):
                            broadened.append(candidate)
                        else:
                            excluded += 1
                if not broadened:
                    if ann_path:
                        # Widen to everything above a minimal embedding similarity
                        broadened = [
                            c
                            for c in scored_candidates
                            if (c.get("law_reranker_score") or 0.0)
                            >= max(0.0, relaxed_rerank_floor * 0.5)
                        ]
                    else:
                        broadened = [
                            candidate
                            for candidate in scored_candidates
                            if int(candidate.get("lexical_score", 0)) >= fallback_min_lexical
                        ]
                if not broadened:
                    broadened = list(scored_candidates)
                routed = [candidate["law_id"] for candidate in broadened[:fallback_max_laws]]
                fallback_mode = "uncertainty_staged"
                uncertainty_selection = {
                    "fallback_min_lexical": fallback_min_lexical,
                    "relaxed_rerank_floor": relaxed_rerank_floor,
                    "candidate_total": len(scored_candidates),
                    "candidate_selected": len(broadened),
                    "candidate_excluded": excluded,
                }
            else:
                # Score-gap window: keep candidates within score_window of the top score,
                # capped at rerank_top_k. Prevents a hard rank cutoff from dropping the
                # correct law when embedding scores are tightly clustered (~0.45-0.60).
                if score_window > 0.0 and selected:
                    top_sel_score = selected[0].get("law_reranker_score")
                    if isinstance(top_sel_score, float):
                        window_floor = top_sel_score - score_window
                        windowed = [
                            c
                            for c in selected
                            if c.get("direct_mention")
                            or (
                                isinstance(c.get("law_reranker_score"), float)
                                and c["law_reranker_score"] >= window_floor
                            )
                        ]
                        routed = [c["law_id"] for c in windowed[:rerank_top_k]]
                    else:
                        routed = [c["law_id"] for c in selected[:rerank_top_k]]
                else:
                    routed = [candidate["law_id"] for candidate in selected[:rerank_top_k]]
        else:
            routed = [candidate["law_id"] for candidate in scored_candidates[:lexical_top_k]]
        routed = [law_id for law_id in routed if (law_id or "").strip()]
        routed = list(dict.fromkeys(routed))

        uncertainty_triggered = bool(fallback_mode and fallback_mode.startswith("uncertainty"))
        routing_confidence = {
            "mode": score_mode,
            "dualpass_enabled": bool(self.settings.law_routing_summary_dualpass_enabled),
            "min_confidence": min_confidence,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "uncertainty_top_score_ceiling": self.settings.law_routing_uncertainty_top_score_ceiling,
            "uncertainty_min_gap": self.settings.law_routing_uncertainty_min_gap,
            "is_uncertain": uncertainty_triggered,
            "selection_mode": (
                "staged_fallback" if fallback_mode == "uncertainty_staged" else "filtered"
            ),
            "selected_law_count": len(routed),
        }
        self._last_routing_diagnostics = {
            "enabled": True,
            "query": query,
            "lexical_candidates": lexical_candidates[:10],
            "scored_candidates": scored_candidates[:10],
            "routed_law_ids": routed,
            "retrieval_fallback": fallback_mode,
            "retrieval_fallback_stage": "stage1_broadened"
            if fallback_mode == "uncertainty_staged"
            else "none",
            "routing_confidence": routing_confidence,
            "uncertainty_selection": uncertainty_selection or {},
        }
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
        self._last_routing_diagnostics.setdefault("retrieval_fallback", None)
        self._last_routing_diagnostics.setdefault("retrieval_fallback_stage", "none")
        if not routed_law_ids:
            return self.retriever.invoke(query)

        law_filter = self._build_law_filter(routed_law_ids)
        stage1_docs = []
        try:
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self._retrieval_k,
                    "filter": law_filter,
                }
            )
            docs = filtered_retriever.invoke(query)
            stage1_docs = docs or []
            should_escalate, escalation_reason = self._should_escalate_to_stage2_fallback(
                query,
                stage1_docs,
                routed_law_ids=routed_law_ids,
                routing_diagnostics=self._last_routing_diagnostics,
            )
            self._last_routing_diagnostics["retrieval_escalation_reason"] = escalation_reason
            if stage1_docs and not should_escalate:
                self._last_routing_diagnostics["retrieval_fallback_stage"] = "stage1_accepted"
                return docs
            self._last_routing_diagnostics["retrieval_fallback"] = "stage1_broadened_low_quality"
            self._last_routing_diagnostics["retrieval_fallback_stage"] = (
                "stage1_broadened_low_quality"
            )
        except Exception as e:
            logger.warning("Filtered retrieval failed (%s); using unfiltered retrieval.", e)
            self._last_routing_diagnostics["retrieval_fallback"] = f"filtered_retrieval_error:{e}"
            self._last_routing_diagnostics["retrieval_fallback_stage"] = "stage1_error"
            self._last_routing_diagnostics["retrieval_escalation_reason"] = "stage1_error"

        if self.settings.law_routing_fallback_unfiltered:
            self._last_routing_diagnostics["retrieval_fallback"] = "uncertainty_unfiltered_stage2"
            self._last_routing_diagnostics["retrieval_fallback_stage"] = "stage2_unfiltered"
            return self.retriever.invoke(query)
        if stage1_docs:
            self._last_routing_diagnostics["retrieval_fallback"] = "stage1_low_quality_kept"
            self._last_routing_diagnostics["retrieval_fallback_stage"] = "stage1_low_quality_kept"
            return stage1_docs
        return []

    def _should_escalate_to_stage2_fallback(
        self,
        query: str,
        docs: list,
        routed_law_ids: list[str] | None = None,
        routing_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Escalate staged fallback when broadened retrieval quality is too weak."""
        routing_diagnostics = routing_diagnostics or self._last_routing_diagnostics or {}
        routing_confidence = routing_diagnostics.get("routing_confidence") or {}
        is_uncertain = bool(routing_confidence.get("is_uncertain"))
        score_gap = routing_confidence.get("score_gap")
        top_route_score = routing_confidence.get("top_score")
        scored_candidates = routing_diagnostics.get("scored_candidates") or []
        candidate_scores = [
            float(candidate.get("law_reranker_score"))
            for candidate in scored_candidates
            if candidate.get("law_reranker_score") is not None
        ]
        routing_concentration = None
        if candidate_scores:
            top_candidate = max(candidate_scores)
            total_candidate = sum(max(score, 0.0) for score in candidate_scores)
            if total_candidate > 0.0:
                routing_concentration = top_candidate / total_candidate
        routed_set = {
            (law_id or "").strip() for law_id in (routed_law_ids or []) if (law_id or "").strip()
        }
        stage1_law_ids = set()
        for doc in docs:
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            law_id = (metadata.get("law_id") or "").strip()
            if law_id:
                stage1_law_ids.add(law_id)
        overlap_count = len(stage1_law_ids & routed_set) if routed_set else None
        min_docs = max(1, int(self.settings.law_routing_stage1_min_docs))
        stage1_quality = {
            "min_docs_threshold": min_docs,
            "doc_count": len(docs),
            "is_uncertain": is_uncertain,
            "routing_concentration": routing_concentration,
            "routing_top_score": top_route_score,
            "routing_score_gap": score_gap,
            "routed_law_count": len(routed_set),
            "stage1_distinct_law_count": len(stage1_law_ids),
            "stage1_routed_overlap_count": overlap_count,
        }
        self._last_routing_diagnostics["stage1_quality"] = stage1_quality
        if len(docs) < min_docs:
            return True, "insufficient_stage1_docs"
        if routed_set and overlap_count == 0:
            return True, "no_routed_law_overlap"
        if not self.reranker:
            if (
                is_uncertain
                and routing_concentration is not None
                and routing_concentration < 0.45
                and self.settings.law_routing_fallback_unfiltered
            ):
                return True, "uncertain_diffuse_routing_without_reranker"
            return False, "accepted_without_reranker"
        top_docs = docs[: min(len(docs), self.settings.retrieval_k)]
        _docs, scores = self._rerank(query, top_docs, top_k=min(3, len(top_docs)))
        if not scores:
            if is_uncertain and self.settings.law_routing_fallback_unfiltered:
                return True, "uncertain_no_stage1_scores"
            return False, "accepted_no_scores"
        top_doc_score = float(scores[0])
        mean_doc_score = float(sum(scores) / len(scores)) if scores else 0.0
        stage1_quality["stage1_top_doc_score"] = top_doc_score
        stage1_quality["stage1_mean_doc_score"] = mean_doc_score
        min_top_score = float(self.settings.law_routing_stage1_min_top_score)
        min_mean_score = float(self.settings.law_routing_stage1_min_mean_score)
        if top_doc_score < min_top_score:
            return True, "low_stage1_top_doc_score"
        if mean_doc_score < min_mean_score:
            return True, "low_stage1_mean_doc_score"

        uncertainty_min_gap = float(self.settings.law_routing_uncertainty_min_gap)
        uncertainty_ceiling = float(self.settings.law_routing_uncertainty_top_score_ceiling)
        weak_top = top_route_score is None or float(top_route_score) <= uncertainty_ceiling
        weak_gap = score_gap is None or float(score_gap) < max(0.01, uncertainty_min_gap * 0.80)
        diffuse_routing = routing_concentration is not None and float(routing_concentration) < 0.45
        if is_uncertain and weak_top and (weak_gap or diffuse_routing):
            if (
                len(stage1_law_ids) > 1
                or top_doc_score < (min_top_score + 0.12)
                or mean_doc_score < (min_mean_score + 0.08)
            ):
                return True, "uncertain_diffuse_stage1_escalation"
        return False, "accepted"

    def _routing_alignment_map(self) -> dict[str, float]:
        """Expose per-law routing alignment scores in [0, 1] for rank fusion."""
        diagnostics = self._last_routing_diagnostics or {}
        scored_candidates = diagnostics.get("scored_candidates") or []
        return compute_routing_alignment(scored_candidates)

    def _apply_reranker_doc_filter(
        self, docs: list, scores: list[float]
    ) -> tuple[list, list[float]]:
        """Filter reranked docs by per-document score with a floor on minimum sources."""
        return filter_reranked_docs(
            docs,
            scores,
            min_doc_score=self.settings.reranker_min_doc_score,
            min_sources=self.settings.reranker_min_sources,
        )

    def _filter_by_law_coherence(self, docs: list, scores: list[float]) -> tuple[list, list[float]]:
        """
        Filter low-confidence sources from non-dominant laws.

        Keeps the dominant law and removes non-dominant laws that have too few
        sources when they are clearly lower-scoring than the dominant law.
        """
        self._last_coherence_diagnostics = {
            "enabled": bool(self.settings.law_coherence_filter_enabled)
        }
        if not docs or not scores or len(docs) != len(scores):
            self._last_coherence_diagnostics.update(
                {"reason": "missing_docs_or_scores", "removed_count": 0}
            )
            return docs, scores
        if len(docs) < 2:
            self._last_coherence_diagnostics.update(
                {"reason": "insufficient_docs", "removed_count": 0}
            )
            return docs, scores

        grouped_indices: dict[str, list[int]] = {}
        for idx, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            law_id = (metadata.get("law_id") or "").strip() or "__unknown__"
            grouped_indices.setdefault(law_id, []).append(idx)

        if len(grouped_indices) <= 1:
            self._last_coherence_diagnostics.update(
                {"reason": "single_law_only", "removed_count": 0}
            )
            return docs, scores

        candidates = []
        for idx, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            candidates.append(
                {
                    "index": idx,
                    "law_id": (metadata.get("law_id") or "").strip(),
                    "score": float(scores[idx]),
                    "cross_references": metadata.get("cross_references") or [],
                }
            )
        law_affinity = build_law_cross_reference_affinity(
            candidates,
            law_ref_to_id=self._law_ref_to_id,
            settings=self.settings,
        )
        decision = build_law_coherence_decision(
            candidates,
            self.settings,
            law_affinity_by_id=law_affinity,
        )
        drop_indices: set[int] = set(decision.get("drop_indices", set()))

        if not drop_indices:
            self._last_coherence_diagnostics.update(
                {
                    "reason": decision.get("reason", "no_laws_met_drop_criteria"),
                    "dominant_law_id": decision.get("dominant_law_id"),
                    "dominant_avg_score": decision.get("dominant_avg_score"),
                    "dominant_max_score": decision.get("dominant_max_score"),
                    "dominant_strength": decision.get("dominant_strength"),
                    "dominant_concentration": decision.get("dominant_concentration"),
                    "cross_reference_affinity": law_affinity,
                    "decisions": decision.get("decisions", []),
                    "removed_count": 0,
                }
            )
            return docs, scores

        filtered_docs = [doc for idx, doc in enumerate(docs) if idx not in drop_indices]
        filtered_scores = [score for idx, score in enumerate(scores) if idx not in drop_indices]
        self._last_coherence_diagnostics.update(
            {
                "reason": decision.get("reason", "filtered"),
                "dominant_law_id": decision.get("dominant_law_id"),
                "dominant_avg_score": decision.get("dominant_avg_score"),
                "dominant_max_score": decision.get("dominant_max_score"),
                "dominant_strength": decision.get("dominant_strength"),
                "dominant_concentration": decision.get("dominant_concentration"),
                "min_sources_floor": decision.get("min_sources_floor"),
                "removed_count": len(drop_indices),
                "cross_reference_affinity": law_affinity,
                "decisions": decision.get("decisions", []),
            }
        )
        logger.debug(
            "Law coherence filter removed %s docs from non-dominant laws (dominant_law=%s)",
            len(drop_indices),
            decision.get("dominant_law_id"),
        )
        return filtered_docs, filtered_scores

    def _apply_law_aware_rank_fusion(
        self, docs: list, scores: list[float]
    ) -> tuple[list, list[float], list[float]]:
        """Apply deterministic law-aware rank fusion after coherence filtering."""
        self._last_rank_fusion_diagnostics = {
            "enabled": bool(self.settings.law_rank_fusion_enabled)
        }
        if (
            not self.settings.law_rank_fusion_enabled
            or not docs
            or not scores
            or len(docs) != len(scores)
        ):
            self._last_rank_fusion_diagnostics.update({"reason": "disabled_or_missing_scores"})
            return docs, scores, scores

        candidates: list[dict[str, Any]] = []
        for idx, (doc, score) in enumerate(zip(docs, scores)):
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            candidates.append(
                {
                    "index": idx,
                    "law_id": (metadata.get("law_id") or "").strip(),
                    "score": float(score),
                    "cross_references": metadata.get("cross_references") or [],
                }
            )

        law_affinity = build_law_cross_reference_affinity(
            candidates,
            law_ref_to_id=self._law_ref_to_id,
            settings=self.settings,
        )
        fused = build_law_aware_rank_fusion(
            candidates,
            self.settings,
            law_affinity_by_id=law_affinity,
            routing_alignment_by_id=self._routing_alignment_map(),
            dominant_context=self._last_coherence_diagnostics,
        )
        ranked_rows = fused.get("ranked") or []
        routing_confidence = (self._last_routing_diagnostics or {}).get("routing_confidence") or {}
        ranked_rows, cap_diag = apply_uncertainty_law_cap(
            ranked_rows,
            fused.get("law_strengths") or {},
            settings=self.settings,
            is_uncertain=bool(routing_confidence.get("is_uncertain")),
        )
        if not ranked_rows:
            self._last_rank_fusion_diagnostics.update({"reason": "empty_ranked_rows"})
            return docs, scores, scores

        # Keep stable, deterministic top-k after fusion.
        top_k = min(self.settings.retrieval_k, len(ranked_rows))
        ranked_rows = ranked_rows[:top_k]
        fused_docs = [docs[int(row["index"])] for row in ranked_rows]
        fused_scores = [float(row.get("fused_score", row.get("score", 0.0))) for row in ranked_rows]
        ce_scores_for_gating = [float(row.get("score", 0.0)) for row in ranked_rows]
        self._last_rank_fusion_diagnostics.update(
            {
                "reason": "applied",
                "fusion": fused.get("diagnostics", {}),
                "law_strengths": fused.get("law_strengths", {}),
                "law_cap": cap_diag,
                "top_rows": [
                    {
                        "law_id": row.get("law_id"),
                        "fused_score": row.get("fused_score"),
                        "ce_score": row.get("score"),
                        "components": row.get("fusion_components"),
                    }
                    for row in ranked_rows[:5]
                ],
            }
        )
        return fused_docs, fused_scores, ce_scores_for_gating

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
                logger.debug(
                    f"Deduplicated {original_count} docs to {len(deduplicated_docs)} before reranking"
                )

        top_score = None
        scores: list[float] = []
        # Rerank if enabled
        if self.reranker and docs:
            docs, scores = self._rerank(retrieval_query, docs, top_k=self.settings.retrieval_k)
            docs, scores = self._apply_reranker_doc_filter(docs, scores)
            if self.settings.law_coherence_filter_enabled:
                docs, scores = self._filter_by_law_coherence(docs, scores)
            docs, fused_scores, gate_scores = self._apply_law_aware_rank_fusion(docs, scores)
            docs, _ = self._attach_editorial_to_provisions(docs, scores=fused_scores)
            scores = gate_scores
            docs, scores = self._prioritize_doc_types(docs, scores, retrieval_query=retrieval_query)
            # Confidence gating stays calibrated on CE reranker scores, not fused scores.
            top_score = scores[0] if scores else None
            if top_score is not None:
                logger.debug(f"Reranked {len(docs)} documents, top score: {top_score:.3f}")
            else:
                logger.debug(f"Reranking returned no scores for {len(docs)} documents")
        elif docs:
            docs, scores = self._attach_editorial_to_provisions(docs, scores=scores)
            docs, scores = self._prioritize_doc_types(docs, scores, retrieval_query=retrieval_query)

        sources = self._extract_sources(docs, include_content=True)
        return sources, top_score, scores

    def should_gate_answer(
        self, top_score: float | None, scores: list[float] | None = None
    ) -> bool:
        """
        Check if answer should be gated due to low confidence.

        Args:
            top_score: Highest reranker score (None if reranking disabled)
            scores: Optional full reranker score list for ambiguity gating

        Returns:
            True if answer should be gated (low confidence)
        """
        return _should_gate_answer_fn(
            top_score,
            scores,
            confidence_threshold=self.settings.reranker_confidence_threshold,
            ambiguity_gating_enabled=self.settings.reranker_ambiguity_gating_enabled,
            ambiguity_top_score_ceiling=self.settings.reranker_ambiguity_top_score_ceiling,
            ambiguity_min_gap=self.settings.reranker_ambiguity_min_gap,
        )

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

    def query(
        self, question: str, chat_history: list[Dict[str, str]] | None = None
    ) -> Dict[str, Any]:
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
        answer = response.content if hasattr(response, "content") else str(response)

        # Return sources without content for consistency
        sources_for_return = [{k: v for k, v in s.items() if k != "content"} for s in sources]

        return {"answer": answer, "sources": sources_for_return}
