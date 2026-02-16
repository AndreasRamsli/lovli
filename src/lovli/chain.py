"""LangChain RAG pipeline for legal question answering."""

import logging
import re
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
from .retrieval_shared import (
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    normalize_law_ref,
    normalize_sigmoid_scores,
    sigmoid,
)

logger = logging.getLogger(__name__)
MAX_QUESTION_LENGTH = 1000

# Query rewriting constants
QUERY_REWRITE_MIN_LENGTH_RATIO = 0.5  # Rewritten query must be at least 50% of original length
QUERY_REWRITE_HISTORY_WINDOW = 4  # Last 4 messages (2 turns) for context
_ROUTING_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_ROUTING_STOPWORDS = {
    "og",
    "som",
    "kan",
    "skal",
    "for",
    "fra",
    "med",
    "ved",
    "til",
    "av",
    "den",
    "det",
    "har",
    "hva",
    "nar",
    "eller",
    "ikke",
    "etter",
}

# Confidence gating response (used by both streaming and non-streaming paths)
GATED_RESPONSE = (
    "Jeg fant ikke et klart svar på spørsmålet ditt i lovtekstene. "
    "Kunne du prøve å omformulere spørsmålet eller være mer spesifikk?"
)
NO_RESULTS_RESPONSE = "Beklager, jeg kunne ikke finne informasjon om dette spørsmålet."


def _sigmoid(x: float) -> float:
    """Apply sigmoid function to map raw logits to [0, 1] probability."""
    return sigmoid(x)


def _tokenize_for_routing(text: str) -> set[str]:
    """Tokenize text for lightweight lexical routing."""
    tokens = set(_ROUTING_TOKEN_RE.findall((text or "").lower()))
    return {token for token in tokens if token not in _ROUTING_STOPWORDS}


def _normalize_law_mention(text: str) -> str:
    """Normalize text for robust law short-name mention checks."""
    normalized = (text or "").lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _infer_doc_type(metadata: Dict[str, Any]) -> str:
    """Infer doc type with backward-compatible fallbacks for legacy payloads."""
    doc_type = (metadata.get("doc_type") or "").strip().lower()
    if doc_type in {"provision", "editorial_note"}:
        return doc_type
    title = (metadata.get("title") or "").strip()
    article_id = (metadata.get("article_id") or "").strip()
    if title == "Untitled Article" or "_art_" in article_id:
        return "editorial_note"
    return "provision"


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
        self._law_catalog_entries: list[dict[str, Any]] = []
        self._law_ref_to_id: dict[str, str] = {}
        self._last_routing_diagnostics: dict[str, Any] = {}
        self._last_coherence_diagnostics: dict[str, Any] = {}
        if self.settings.law_routing_enabled:
            try:
                self._law_catalog = load_catalog(Path(self.settings.law_catalog_path))
                self._law_catalog_entries = self._build_routing_entries(self._law_catalog)
                self._law_ref_to_id = {
                    normalize_law_ref(item.get("law_ref") or ""): (item.get("law_id") or "").strip()
                    for item in self._law_catalog
                    if normalize_law_ref(item.get("law_ref") or "") and (item.get("law_id") or "").strip()
                }
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

    def _build_routing_entries(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Precompute routing text/tokens for fast hybrid law routing."""
        entries: list[dict[str, Any]] = []
        for item in catalog:
            law_id = (item.get("law_id") or "").strip()
            if not law_id:
                continue
            title = (item.get("law_title") or "").strip()
            short_name = (item.get("law_short_name") or "").strip()
            summary = (item.get("summary") or "").strip()
            law_ref = (item.get("law_ref") or "").strip()
            legal_area = (item.get("legal_area") or "").strip()
            chapter_titles = item.get("chapter_titles") or []
            if not isinstance(chapter_titles, list):
                chapter_titles = [str(chapter_titles)]
            chapter_titles_text = " ".join(
                chapter_title.strip()
                for chapter_title in chapter_titles[:5]
                if isinstance(chapter_title, str) and chapter_title.strip()
            )
            routing_text = " ".join(
                part
                for part in [title, short_name, summary, law_ref, legal_area, chapter_titles_text]
                if part
            )
            entries.append(
                {
                    "law_id": law_id,
                    "law_title": title,
                    "law_short_name": short_name,
                    "routing_text": routing_text,
                    "routing_tokens": _tokenize_for_routing(routing_text),
                    "short_name_normalized": _normalize_law_mention(short_name),
                    "law_title_normalized": _normalize_law_mention(title),
                }
            )
        return entries

    def _score_law_candidates_lexical(self, query: str) -> list[dict[str, Any]]:
        """Score catalog entries lexically before optional reranker-based law scoring."""
        query_tokens = _tokenize_for_routing(query)
        if not query_tokens:
            return []
        query_norm = _normalize_law_mention(query)

        scored: list[dict[str, Any]] = []
        for entry in self._law_catalog_entries:
            overlap = len(query_tokens & entry["routing_tokens"])
            short_name = entry.get("short_name_normalized") or ""
            law_title = entry.get("law_title_normalized") or ""

            direct_mention = False
            if short_name and short_name in query_norm:
                direct_mention = True
                overlap += 5
            elif law_title and law_title in query_norm:
                direct_mention = True
                overlap += 5

            if overlap < self.settings.law_routing_min_token_overlap:
                continue

            scored.append(
                {
                    "law_id": entry["law_id"],
                    "law_title": entry.get("law_title", ""),
                    "law_short_name": entry.get("law_short_name", ""),
                    "routing_text": entry.get("routing_text", ""),
                    "lexical_score": overlap,
                    "direct_mention": direct_mention,
                }
            )

        scored.sort(
            key=lambda item: (item["direct_mention"], item["lexical_score"]),
            reverse=True,
        )
        return scored[: self.settings.law_routing_prefilter_k]

    def _score_law_candidates_reranker(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply law-level reranker scoring to lexical candidates."""
        if not candidates:
            return candidates
        if not self.reranker:
            for candidate in candidates:
                candidate["law_reranker_score"] = None
            return candidates

        pairs = [
            [query, candidate.get("routing_text", "") or candidate.get("law_title", "")]
            for candidate in candidates
        ]
        try:
            raw_scores = self.reranker.predict(pairs)
            law_scores = normalize_sigmoid_scores(raw_scores)
        except Exception as exc:
            logger.warning("Law-level reranker scoring failed; falling back to lexical routing (%s)", exc)
            for candidate in candidates:
                candidate["law_reranker_score"] = None
            return candidates

        reranked = []
        for candidate, score in zip(candidates, law_scores):
            item = dict(candidate)
            item["law_reranker_score"] = score
            reranked.append(item)

        # Direct law mentions are trusted strongly: keep them high.
        reranked.sort(
            key=lambda item: (
                item.get("law_reranker_score", 0.0),
                item.get("lexical_score", 0),
            ),
            reverse=True,
        )
        return reranked

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
            editorial_docs = self._fetch_editorial_for_chapters(law_chapter_pairs, per_chapter_cap=2)

        for editorial_doc in editorial_docs:
            e_meta = editorial_doc.metadata if hasattr(editorial_doc, "metadata") else {}
            note_payload = _editorial_note_payload(editorial_doc)
            law_id = (e_meta.get("law_id") or "").strip()
            linked_provision_id = (e_meta.get("linked_provision_id") or "").strip()
            chapter_id = (e_meta.get("chapter_id") or "").strip()
            if law_id and linked_provision_id:
                editorial_by_provision.setdefault((law_id, linked_provision_id), []).append(note_payload)
            elif law_id and chapter_id:
                editorial_by_chapter.setdefault((law_id, chapter_id), []).append(note_payload)

        chapter_to_provision_keys: Dict[tuple[str, str], list[tuple[str, str]]] = {}
        for provision_doc in provisions:
            p_meta = provision_doc.metadata if hasattr(provision_doc, "metadata") else {}
            law_id = (p_meta.get("law_id") or "").strip()
            article_id = (p_meta.get("article_id") or "").strip()
            chapter_id = (p_meta.get("chapter_id") or "").strip()
            if law_id and chapter_id and article_id:
                chapter_to_provision_keys.setdefault((law_id, chapter_id), []).append((law_id, article_id))

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
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            if isinstance(raw_scores, (int, float)):
                raw_scores = [raw_scores]
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
        scores = normalize_sigmoid_scores(raw_scores)

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
        """Route query to likely laws using hybrid lexical + law-level reranker scoring."""
        if not self.settings.law_routing_enabled or not self._law_catalog_entries:
            self._last_routing_diagnostics = {
                "enabled": bool(self.settings.law_routing_enabled),
                "reason": "routing_disabled_or_empty_catalog",
                "routed_law_ids": [],
            }
            return []
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
        lexical_top_k = self.settings.law_routing_max_candidates
        fallback_max_laws = self.settings.law_routing_fallback_max_laws

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
            if top_score is not None and second_score is not None:
                score_gap = top_score - second_score
                score_mode = "reranker"

        def _is_uncertain(candidates: list[dict[str, Any]]) -> bool:
            if not candidates or not self.reranker:
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

        if self.reranker:
            selected = [
                candidate
                for candidate in scored_candidates
                if candidate.get("direct_mention")
                or (candidate.get("law_reranker_score") is not None and candidate.get("law_reranker_score", 0.0) >= min_confidence)
            ]
            if not selected:
                selected = scored_candidates[: max(lexical_top_k, rerank_top_k)]
            if _is_uncertain(scored_candidates):
                if self.settings.law_routing_fallback_unfiltered:
                    routed = []
                    fallback_mode = "uncertainty_unfiltered"
                else:
                    broadened = scored_candidates[:fallback_max_laws]
                    routed = [candidate["law_id"] for candidate in broadened]
                    fallback_mode = "uncertainty_broadened"
            else:
                routed = [candidate["law_id"] for candidate in selected[:rerank_top_k]]
                fallback_mode = None
        else:
            routed = [candidate["law_id"] for candidate in scored_candidates[:lexical_top_k]]
            fallback_mode = None

        uncertainty_triggered = bool(fallback_mode and fallback_mode.startswith("uncertainty"))
        routing_confidence = {
            "mode": score_mode,
            "min_confidence": min_confidence,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "uncertainty_top_score_ceiling": self.settings.law_routing_uncertainty_top_score_ceiling,
            "uncertainty_min_gap": self.settings.law_routing_uncertainty_min_gap,
            "is_uncertain": uncertainty_triggered,
            "selection_mode": (
                "unfiltered_fallback"
                if fallback_mode == "uncertainty_unfiltered"
                else "broadened_fallback"
                if fallback_mode == "uncertainty_broadened"
                else "filtered"
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
            "routing_confidence": routing_confidence,
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
            self._last_routing_diagnostics["retrieval_fallback"] = "empty_filtered_results"
        except Exception as e:
            logger.warning("Filtered retrieval failed (%s); using unfiltered retrieval.", e)
            self._last_routing_diagnostics["retrieval_fallback"] = f"filtered_retrieval_error:{e}"

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

    def _filter_by_law_coherence(
        self, docs: list, scores: list[float]
    ) -> tuple[list, list[float]]:
        """
        Filter low-confidence sources from non-dominant laws.

        Keeps the dominant law and removes non-dominant laws that have too few
        sources when they are clearly lower-scoring than the dominant law.
        """
        self._last_coherence_diagnostics = {"enabled": bool(self.settings.law_coherence_filter_enabled)}
        if not docs or not scores or len(docs) != len(scores):
            self._last_coherence_diagnostics.update({"reason": "missing_docs_or_scores", "removed_count": 0})
            return docs, scores
        if len(docs) < 2:
            self._last_coherence_diagnostics.update({"reason": "insufficient_docs", "removed_count": 0})
            return docs, scores

        grouped_indices: dict[str, list[int]] = {}
        for idx, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            law_id = (metadata.get("law_id") or "").strip() or "__unknown__"
            grouped_indices.setdefault(law_id, []).append(idx)

        if len(grouped_indices) <= 1:
            self._last_coherence_diagnostics.update({"reason": "single_law_only", "removed_count": 0})
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
            if self.settings.law_coherence_filter_enabled:
                docs, scores = self._filter_by_law_coherence(docs, scores)
            docs, scores = self._attach_editorial_to_provisions(docs, scores=scores)
            docs, scores = self._prioritize_doc_types(docs, scores, retrieval_query=retrieval_query)
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
