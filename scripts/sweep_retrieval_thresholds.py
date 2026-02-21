#!/usr/bin/env python3
"""
Run a lightweight retrieval quality sweep over threshold combinations.

This evaluates retrieval/reranking behavior only (no answer generation), so
you can quickly pick robust defaults before full LangSmith runs.
"""

import itertools
import json
import logging
import os
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain  # noqa: E402
from lovli.config import get_settings  # noqa: E402
from lovli.eval import infer_negative_type, validate_questions  # noqa: E402
from lovli.scoring import (  # noqa: E402
    _infer_doc_type,
    apply_uncertainty_law_cap,
    build_law_aware_rank_fusion,
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    matches_expected_source,
    normalize_sigmoid_scores,
)
from lovli.profiles import apply_trust_profile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Frozen core subset for stable regressions across tuning runs.
CORE_QUESTION_IDS = {
    "q001",
    "q002",
    "q003",
    "q004",
    "q005",
    "q006",
    "q007",
    "q008",
    "q009",
    "q010",
    "q011",
    "q012",
    "q013",
    "q014",
    "q015",
    "q016",
    "q017",
    "q018",
    "q019",
    "q020",
    "q021",
    "q022",
    "q031",
    "q036",
    "q037",
    "q038",
    "q039",
    "q040",
    "q041",
    "q042",
    "q045",
    "q046",
    "q047",
    "q048",
    "q049",
    "q050",
}


def matches_expected(cited_id: str, expected_set: set[str]) -> bool:
    """Prefix-match cited article IDs against expected IDs."""
    return any(cited_id.startswith(exp) for exp in expected_set)


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with configurable default."""
    return (numerator / denominator) if denominator else default


def _article_family(article_id: str) -> str:
    """Normalize nested article IDs to a paragraph-family key."""
    token = (article_id or "").strip()
    if not token:
        return ""
    for marker in ("-ledd-", "-nummer-", "-punkt-"):
        if marker in token:
            return token.split(marker)[0]
    return token


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _safe_git_commit() -> str:
    """Best-effort current git commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT_DIR),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    """Compute sha256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _precompute_cache_key(
    questions_sha256: str,
    git_commit: str,
    qdrant_collection: str,
    max_k_initial: int,
    fallback_unfiltered: bool,
    reranker_ctx_enabled: bool,
    dualpass_enabled: bool,
) -> str:
    """Deterministic cache key for precomputed candidates. Invalidation on any input change."""
    blob = (
        f"{questions_sha256}|{git_commit}|{qdrant_collection}|{max_k_initial}|"
        f"{fallback_unfiltered}|{reranker_ctx_enabled}|{dualpass_enabled}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def classify_case_type(row: dict) -> str:
    """Infer or return declared case type for segmented reporting."""
    case_type = row.get("case_type")
    if case_type in {"single_article", "multi_article", "negative"}:
        return case_type
    expected = row.get("expected_articles", [])
    if not expected:
        return "negative"
    if len(expected) > 1:
        return "multi_article"
    return "single_article"


def _env_flag(name: str, default: bool) -> bool:
    """Parse bool-like env values such as 1/0, true/false, yes/no."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _has_attached_editorial(provision_rows: list[dict]) -> bool:
    """Return True when any kept provision already carries editorial payload."""
    for candidate in provision_rows or []:
        if bool(candidate.get("has_editorial_notes")):
            return True
        if (candidate.get("editorial_notes_count") or 0) > 0:
            return True
    return False


def _compute_balanced_score(metrics: dict) -> float:
    """
    Composite score balancing positive accuracy and negative guardrails.

    Weights prioritize answer quality while still penalizing contamination/noise.
    """
    accuracy_component = (
        (0.16 * metrics.get("recall_at_k", 0.0))
        + (0.20 * metrics.get("coverage_weighted_positive_recall", 0.0))
        + (0.14 * metrics.get("mean_expected_coverage", 0.0))
        + (0.10 * metrics.get("citation_precision", 0.0))
        + (0.05 * metrics.get("multi_article_recall_at_k", 0.0))
    )
    abstention_component = (
        (0.17 * metrics.get("negative_success", 0.0))
        + (0.08 * metrics.get("negative_offtopic_legal_success", 0.0))
        + (0.05 * metrics.get("negative_offtopic_nonlegal_success", 0.0))
    )
    penalties = (
        (0.08 * metrics.get("law_contamination_rate", 0.0))
        + (0.07 * metrics.get("unexpected_citation_rate", 0.0))
        + (0.10 * metrics.get("false_positive_gating_rate", 0.0))
        + (0.05 * metrics.get("source_boundary_mismatch_at_k", 0.0))
    )
    return accuracy_component + abstention_component - penalties


def _is_profile_default_row(row: dict, profile_defaults: dict) -> bool:
    """Return True when a sweep row exactly matches active profile defaults."""
    for key, expected in profile_defaults.items():
        if row.get(key) != expected:
            return False
    return True


def _with_default_value(base_values: list, default_value):
    """Return sorted unique values ensuring profile default is present."""
    merged = list(base_values)
    if default_value not in merged:
        merged.append(default_value)
    return sorted(set(merged))


def _apply_law_coherence_filter_candidates(
    candidates: list[dict],
    settings,
    law_ref_to_id: dict[str, str] | None = None,
) -> tuple[list[dict], int, dict]:
    """Apply runtime-equivalent law coherence filtering on scored candidate dicts."""
    if not settings.law_coherence_filter_enabled:
        return candidates, 0, {"reason": "disabled"}
    affinity = build_law_cross_reference_affinity(
        candidates,
        law_ref_to_id=law_ref_to_id,
        settings=settings,
    )
    decision = build_law_coherence_decision(
        candidates,
        settings,
        law_affinity_by_id=affinity,
    )
    drop_indices = set(decision.get("drop_indices", set()))
    if not drop_indices:
        return candidates, 0, decision
    filtered = [item for idx, item in enumerate(candidates) if idx not in drop_indices]
    return filtered, len(drop_indices), decision


def _build_routing_alignment_map(routing_diagnostics: dict | None) -> dict[str, float]:
    """Build law routing-alignment map in [0, 1] from routing diagnostics."""
    diagnostics = routing_diagnostics or {}
    scored_candidates = diagnostics.get("scored_candidates") or []
    alignment: dict[str, float] = {}
    for candidate in scored_candidates:
        law_id = (candidate.get("law_id") or "").strip()
        if not law_id:
            continue
        score = candidate.get("law_reranker_score")
        lexical = float(candidate.get("lexical_score", 0.0))
        if score is None:
            score = min(1.0, lexical / 10.0)
        alignment[law_id] = max(0.0, min(1.0, float(score)))
    return alignment


def _process_single_question(
    chain: LegalRAGChain,
    row: dict,
    max_k_initial: int,
) -> dict:
    """Process a single question and return cached candidate dict."""
    query = row["question"]
    routed_law_ids = chain._route_law_ids(query)
    route_diag_before = dict(getattr(chain, "_last_routing_diagnostics", {}) or {})
    docs = chain._invoke_retriever(query, routed_law_ids=routed_law_ids)
    route_diag_after = dict(
        getattr(chain, "_last_routing_diagnostics", route_diag_before) or route_diag_before
    )

    dedup_docs = []
    seen_keys = set()
    for doc in docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        key = (metadata.get("law_id"), metadata.get("article_id"))
        if metadata.get("article_id") and key not in seen_keys:
            seen_keys.add(key)
            dedup_docs.append(doc)
        elif not metadata.get("article_id"):
            dedup_docs.append(doc)
    docs = dedup_docs[:max_k_initial]

    candidates: list[dict] = []
    if docs:
        pairs = [[query, chain._build_reranker_document_text(doc)] for doc in docs]
        if chain.reranker:
            raw_scores = chain.reranker.predict(pairs, batch_size=32)
        else:
            raw_scores = [1.0] * len(docs)
        normalized = normalize_sigmoid_scores(raw_scores)
        for doc, score in zip(docs, normalized):
            metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            candidates.append(
                {
                    "law_id": metadata.get("law_id", ""),
                    "article_id": metadata.get("article_id", ""),
                    "chapter_id": metadata.get("chapter_id", ""),
                    "doc_type": _infer_doc_type(metadata),
                    "has_editorial_notes": bool(metadata.get("editorial_notes")),
                    "editorial_notes_count": len(metadata.get("editorial_notes") or []),
                    "cross_references": metadata.get("cross_references") or [],
                    "score": score,
                }
            )

    return {
        "id": row.get("id"),
        "question": query,
        "expected_articles": row.get("expected_articles", []),
        "expected_sources": row.get("expected_sources", []),
        "expects_editorial_context": bool(row.get("expects_editorial_context", False)),
        "case_type": classify_case_type(row),
        "negative_type": infer_negative_type(row),
        "is_core": row.get("id") in CORE_QUESTION_IDS,
        "routing_is_uncertain": bool(
            ((route_diag_after.get("routing_confidence") or {}).get("is_uncertain"))
        ),
        "routing_alignment_by_law": _build_routing_alignment_map(route_diag_after),
        "retrieval_fallback_stage": route_diag_after.get("retrieval_fallback_stage"),
        "retrieval_fallback": route_diag_after.get("retrieval_fallback"),
        "candidates": candidates,
    }


def precompute_candidates(
    chain: LegalRAGChain,
    questions: list[dict],
    max_k_initial: int,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[dict]:
    """Run retrieval/reranker once per question and cache candidates for offline sweeping.

    Args:
        chain: LegalRAGChain instance
        questions: List of question dicts
        max_k_initial: Maximum initial retrieval k
        parallel: If True, use parallel processing (requires fresh chain per thread)
        max_workers: Number of parallel workers (default 4)
    """
    chain.retriever = chain.vectorstore.as_retriever(search_kwargs={"k": max_k_initial})

    if parallel:
        logger.info("Using parallel precomputation with %d workers", max_workers)
        cached: list[dict | None] = [None] * len(questions)

        def process_with_chain(idx: int, row: dict) -> tuple[int, dict]:
            new_chain = LegalRAGChain(settings=chain.settings)
            new_chain.retriever = new_chain.vectorstore.as_retriever(
                search_kwargs={"k": max_k_initial}
            )
            result = _process_single_question(new_chain, row, max_k_initial)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_with_chain, i, row): i for i, row in enumerate(questions)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                cached[idx] = result
                if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
                    logger.info("Precomputed %d/%d questions", idx + 1, len(questions))
    else:
        cached = []
        for i, row in enumerate(questions):
            result = _process_single_question(chain, row, max_k_initial)
            cached.append(result)
            if (i + 1) % 10 == 0 or i == len(questions) - 1:
                logger.info("Precomputed %d/%d questions", i + 1, len(questions))

    return [c for c in cached if c is not None]


def evaluate_combo(
    chain: LegalRAGChain,
    cached_candidates: list[dict],
    retrieval_k_initial: int,
    confidence_threshold: float,
    min_doc_score: float,
    ambiguity_min_gap: float,
    top_score_ceiling: float,
) -> dict:
    """Compute retrieval metrics for one config combo using cached candidates."""
    calibration_bin_count = 5
    calibration_bins = [
        {
            "low": i / calibration_bin_count,
            "high": (i + 1) / calibration_bin_count,
            "total": 0,
            "hits": 0,
        }
        for i in range(calibration_bin_count)
    ]
    retrieval_hits = 0
    retrieval_total = 0
    expected_coverage_total = 0.0
    expected_coverage_count = 0
    citation_precision_total = 0.0
    citation_precision_count = 0
    unexpected_citation_rate_total = 0.0
    unexpected_citation_rate_count = 0
    ambiguity_total = 0
    ambiguity_clean = 0
    editorial_expected_total = 0
    editorial_success = 0
    core_editorial_expected_total = 0
    core_editorial_success = 0
    non_editorial_expected_total = 0
    non_editorial_clean = 0
    core_non_editorial_expected_total = 0
    core_non_editorial_clean = 0
    law_contamination_total = 0
    law_contamination_cases = 0
    law_coherence_filtered_count = 0
    false_positive_gating_count = 0
    false_positive_confidence_gating_count = 0
    false_positive_ambiguity_gating_count = 0
    confidence_gating_count = 0
    ambiguity_gating_count = 0
    confidence_threshold_crossings = 0
    ambiguity_threshold_crossings = 0
    min_sources_floor_trigger_count = 0
    scored_query_count = 0
    positive_total = 0
    weighted_positive_recall_numerator = 0.0
    weighted_positive_recall_denominator = 0.0
    recall_at_1_hits = 0
    recall_at_3_hits = 0
    recall_at_5_hits = 0
    reciprocal_rank_sum = 0.0
    mrr_count = 0
    boundary_items_total = 0
    boundary_wrong_law_count = 0
    boundary_wrong_article_same_law_count = 0
    boundary_family_proxy_count = 0
    top_scores: list[float] = []
    routing_uncertain_count = 0
    fallback_stage1_accepted_count = 0
    fallback_stage1_low_quality_kept_count = 0
    fallback_stage2_unfiltered_count = 0
    fallback_stage1_error_count = 0
    final_k = chain.settings.retrieval_k
    min_sources = chain.settings.reranker_min_sources
    segment = {
        "single_article": {"hits": 0, "total": 0},
        "multi_article": {"hits": 0, "total": 0},
        "negative": {"clean": 0, "total": 0},
        "negative_ambiguity": {"clean": 0, "total": 0},
        "negative_offtopic_legal": {"clean": 0, "total": 0},
        "negative_offtopic_nonlegal": {"clean": 0, "total": 0},
    }
    core_segment = {
        "single_article": {"hits": 0, "total": 0},
        "multi_article": {"hits": 0, "total": 0},
        "negative": {"clean": 0, "total": 0},
        "negative_ambiguity": {"clean": 0, "total": 0},
        "negative_offtopic_legal": {"clean": 0, "total": 0},
        "negative_offtopic_nonlegal": {"clean": 0, "total": 0},
    }

    for row in cached_candidates:
        expected = set(row.get("expected_articles", []))
        expected_sources = row.get("expected_sources", []) or []
        candidates = row.get("candidates", [])
        question = row.get("question", "")
        case_type = row.get("case_type", "single_article")
        negative_type = row.get("negative_type", "unknown")
        is_core = bool(row.get("is_core"))
        expects_editorial_context = bool(row.get("expects_editorial_context", False))
        if bool(row.get("routing_is_uncertain")):
            routing_uncertain_count += 1
        fallback_stage = (row.get("retrieval_fallback_stage") or "").strip()
        if fallback_stage == "stage1_accepted":
            fallback_stage1_accepted_count += 1
        elif fallback_stage == "stage1_low_quality_kept":
            fallback_stage1_low_quality_kept_count += 1
        elif fallback_stage == "stage2_unfiltered":
            fallback_stage2_unfiltered_count += 1
        elif fallback_stage == "stage1_error":
            fallback_stage1_error_count += 1

        # Simulate retrieval_k_initial by truncating pre-rerank candidates.
        subset = candidates[:retrieval_k_initial]
        ranked = sorted(subset, key=lambda x: x["score"], reverse=True)[:final_k]

        # Simulate per-doc score filtering + floor.
        kept = [c for c in ranked if c["score"] >= min_doc_score]
        floor_size = min(min_sources, len(ranked))
        if len(kept) < floor_size:
            kept = ranked[: min(min_sources, len(ranked))]
            min_sources_floor_trigger_count += 1
        # Attachment model: editorial candidates are attached to provision rows,
        # not appended as standalone documents.
        provision_kept = [c for c in kept if c.get("doc_type") != "editorial_note"]
        provision_kept, coherence_dropped, coherence_decision = (
            _apply_law_coherence_filter_candidates(
                provision_kept,
                chain.settings,
                law_ref_to_id=getattr(chain, "_law_ref_to_id", {}) or {},
            )
        )
        law_coherence_filtered_count += coherence_dropped
        if provision_kept:
            fusion_candidates = [
                {
                    "index": idx,
                    "law_id": (candidate.get("law_id") or "").strip(),
                    "score": float(candidate.get("score", 0.0)),
                    "cross_references": candidate.get("cross_references") or [],
                }
                for idx, candidate in enumerate(provision_kept)
            ]
            law_affinity = build_law_cross_reference_affinity(
                fusion_candidates,
                law_ref_to_id=getattr(chain, "_law_ref_to_id", {}) or {},
                settings=chain.settings,
            )
            fused = build_law_aware_rank_fusion(
                fusion_candidates,
                chain.settings,
                law_affinity_by_id=law_affinity,
                routing_alignment_by_id=row.get("routing_alignment_by_law") or {},
                dominant_context=coherence_decision,
            )
            ranked_rows = fused.get("ranked") or []
            ranked_rows, _cap_diag = apply_uncertainty_law_cap(
                ranked_rows,
                fused.get("law_strengths") or {},
                settings=chain.settings,
                is_uncertain=bool(row.get("routing_is_uncertain")),
            )
            top_k = min(final_k, len(ranked_rows))
            ranked_rows = ranked_rows[:top_k]
            provision_kept = [provision_kept[int(item["index"])] for item in ranked_rows]
            # Keep CE score stream for gate semantics.
            scores = [float(item.get("score", 0.0)) for item in ranked_rows]
        else:
            scores = []
        cited_ids = [c["article_id"] for c in provision_kept if c.get("article_id")]
        cited_sources = [
            {"law_id": c.get("law_id", ""), "article_id": c.get("article_id", "")}
            for c in provision_kept
            if c.get("article_id")
        ]
        cited_editorial = _has_attached_editorial(provision_kept)
        top_score = scores[0] if scores else None
        if provision_kept:
            law_contamination_total += 1
            distinct_laws = {c.get("law_id", "") for c in provision_kept if c.get("law_id")}
            if len(distinct_laws) > 1:
                law_contamination_cases += 1

        if top_score is not None:
            top_scores.append(top_score)
            scored_query_count += 1
            if top_score < confidence_threshold:
                confidence_threshold_crossings += 1
            if (
                len(scores) >= 2
                and scores[0] <= top_score_ceiling
                and (scores[0] - scores[1]) < ambiguity_min_gap
            ):
                ambiguity_threshold_crossings += 1

        confidence_gate_hit = bool(top_score is not None and top_score < confidence_threshold)
        ambiguity_gate_hit = bool(
            not confidence_gate_hit
            and chain.settings.reranker_ambiguity_gating_enabled
            and len(scores) >= 2
            and scores[0] <= top_score_ceiling
            and (scores[0] - scores[1]) < ambiguity_min_gap
        )
        is_gated = confidence_gate_hit or ambiguity_gate_hit
        if confidence_gate_hit:
            confidence_gating_count += 1
        elif ambiguity_gate_hit:
            ambiguity_gating_count += 1

        if expected:
            retrieval_total += 1
            positive_total += 1
            expected_law_ids = set()
            expected_article_ids = set()
            if expected_sources:
                expected_law_ids = {
                    (item.get("law_id") or "").strip()
                    for item in expected_sources
                    if (item.get("law_id") or "").strip()
                }
                expected_article_ids = {
                    (item.get("article_id") or "").strip()
                    for item in expected_sources
                    if (item.get("article_id") or "").strip()
                }
            else:
                expected_article_ids = {item.strip() for item in expected if item and item.strip()}

            # Rank-sensitive retrieval metrics on post-filter top-k.
            def _is_relevant_at_position(cited_source: dict[str, str]) -> bool:
                if expected_sources:
                    return any(
                        matches_expected_source(cited_source, exp) for exp in expected_sources
                    )
                cited_article = (cited_source.get("article_id") or "").strip()
                return any(cited_article.startswith(exp_id) for exp_id in expected_article_ids)

            hit_at_1 = False
            hit_at_3 = False
            hit_at_5 = False
            first_relevant_rank = None
            for rank_idx, cited_source in enumerate(cited_sources, start=1):
                relevant = _is_relevant_at_position(cited_source)
                if relevant:
                    if first_relevant_rank is None:
                        first_relevant_rank = rank_idx
                    if rank_idx <= 1:
                        hit_at_1 = True
                    if rank_idx <= 3:
                        hit_at_3 = True
                    if rank_idx <= 5:
                        hit_at_5 = True
            recall_at_1_hits += int(hit_at_1)
            recall_at_3_hits += int(hit_at_3)
            recall_at_5_hits += int(hit_at_5)
            mrr_count += 1
            if first_relevant_rank is not None and first_relevant_rank <= 5:
                reciprocal_rank_sum += 1.0 / first_relevant_rank

            # DRM-analog boundary diagnostics across retrieved items.
            expected_family_ids = {_article_family(item) for item in expected_article_ids if item}
            for cited in cited_sources:
                cited_law = (cited.get("law_id") or "").strip()
                cited_article = (cited.get("article_id") or "").strip()
                if not cited_article:
                    continue
                boundary_items_total += 1
                law_matches = (not expected_law_ids) or (cited_law in expected_law_ids)
                source_matches = _is_relevant_at_position(cited)
                if not law_matches:
                    boundary_wrong_law_count += 1
                    continue
                if source_matches:
                    continue
                boundary_wrong_article_same_law_count += 1
                if _article_family(cited_article) in expected_family_ids:
                    boundary_family_proxy_count += 1

            if expected_sources:
                matched_expected_pairs = set()
                matched_citations = 0
                for cited in cited_sources:
                    hit = False
                    for exp in expected_sources:
                        if matches_expected_source(cited, exp):
                            pair_key = (exp.get("law_id", ""), exp.get("article_id", ""))
                            matched_expected_pairs.add(pair_key)
                            hit = True
                    if hit:
                        matched_citations += 1
                coverage = len(matched_expected_pairs) / len(expected_sources)
                matched = len(matched_expected_pairs) > 0
                precision = (matched_citations / len(cited_sources)) if cited_sources else 0.0
                unexpected_rate = (
                    (len(cited_sources) - matched_citations) / len(cited_sources)
                    if cited_sources
                    else 0.0
                )
            else:
                found_expected = set()
                matched_citations = 0
                for cid in cited_ids:
                    hit = False
                    for exp in expected:
                        if cid.startswith(exp):
                            found_expected.add(exp)
                            hit = True
                    if hit:
                        matched_citations += 1
                coverage = len(found_expected) / len(expected) if expected else 0.0
                matched = len(found_expected) > 0
                precision = (matched_citations / len(cited_ids)) if cited_ids else 0.0
                unexpected_rate = (
                    (len(cited_ids) - matched_citations) / len(cited_ids) if cited_ids else 0.0
                )

            if matched:
                retrieval_hits += 1
            # Coverage-weighted positive recall is less binary than recall_at_k.
            weighted_positive_recall_numerator += coverage
            weighted_positive_recall_denominator += 1.0
            expected_coverage_total += coverage
            expected_coverage_count += 1
            citation_precision_total += precision
            citation_precision_count += 1
            unexpected_citation_rate_total += unexpected_rate
            unexpected_citation_rate_count += 1
            if case_type in {"single_article", "multi_article"}:
                segment[case_type]["total"] += 1
                if matched:
                    segment[case_type]["hits"] += 1
                if is_core:
                    core_segment[case_type]["total"] += 1
                    if matched:
                        core_segment[case_type]["hits"] += 1
            if top_score is not None:
                bin_idx = min(int(top_score * calibration_bin_count), calibration_bin_count - 1)
                calibration_bins[bin_idx]["total"] += 1
                if matched:
                    calibration_bins[bin_idx]["hits"] += 1
        else:
            # Ambiguous / off-topic guardrail: ideally no sources, or gated.
            ambiguity_total += 1
            if is_gated or not cited_ids:
                ambiguity_clean += 1
                segment["negative"]["clean"] += 1
                segment_key = f"negative_{negative_type}"
                if segment_key in segment:
                    segment[segment_key]["clean"] += 1
                if is_core:
                    core_segment["negative"]["clean"] += 1
                    if segment_key in core_segment:
                        core_segment[segment_key]["clean"] += 1
            segment["negative"]["total"] += 1
            segment_key = f"negative_{negative_type}"
            if segment_key in segment:
                segment[segment_key]["total"] += 1
            if is_core:
                core_segment["negative"]["total"] += 1
                if segment_key in core_segment:
                    core_segment[segment_key]["total"] += 1

        if expected:
            if is_gated:
                false_positive_gating_count += 1
                if confidence_gate_hit:
                    false_positive_confidence_gating_count += 1
                elif ambiguity_gate_hit:
                    false_positive_ambiguity_gating_count += 1

        if expects_editorial_context:
            editorial_expected_total += 1
            if cited_editorial:
                editorial_success += 1
            if is_core:
                core_editorial_expected_total += 1
                if cited_editorial:
                    core_editorial_success += 1
        else:
            non_editorial_expected_total += 1
            if not cited_editorial:
                non_editorial_clean += 1
            if is_core:
                core_non_editorial_expected_total += 1
                if not cited_editorial:
                    core_non_editorial_clean += 1

    recall_at_k = retrieval_hits / retrieval_total if retrieval_total else 0.0
    ambiguity_success = ambiguity_clean / ambiguity_total if ambiguity_total else 1.0
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    single_recall = (
        segment["single_article"]["hits"] / segment["single_article"]["total"]
        if segment["single_article"]["total"]
        else 0.0
    )
    multi_recall = (
        segment["multi_article"]["hits"] / segment["multi_article"]["total"]
        if segment["multi_article"]["total"]
        else 0.0
    )
    negative_success = (
        segment["negative"]["clean"] / segment["negative"]["total"]
        if segment["negative"]["total"]
        else 1.0
    )
    negative_ambiguity_success = (
        segment["negative_ambiguity"]["clean"] / segment["negative_ambiguity"]["total"]
        if segment["negative_ambiguity"]["total"]
        else 1.0
    )
    negative_offtopic_legal_success = (
        segment["negative_offtopic_legal"]["clean"] / segment["negative_offtopic_legal"]["total"]
        if segment["negative_offtopic_legal"]["total"]
        else 1.0
    )
    negative_offtopic_nonlegal_success = (
        segment["negative_offtopic_nonlegal"]["clean"]
        / segment["negative_offtopic_nonlegal"]["total"]
        if segment["negative_offtopic_nonlegal"]["total"]
        else 1.0
    )
    core_single_recall = (
        core_segment["single_article"]["hits"] / core_segment["single_article"]["total"]
        if core_segment["single_article"]["total"]
        else 0.0
    )
    core_multi_recall = (
        core_segment["multi_article"]["hits"] / core_segment["multi_article"]["total"]
        if core_segment["multi_article"]["total"]
        else 0.0
    )
    core_negative_success = (
        core_segment["negative"]["clean"] / core_segment["negative"]["total"]
        if core_segment["negative"]["total"]
        else 1.0
    )
    core_negative_ambiguity_success = (
        core_segment["negative_ambiguity"]["clean"] / core_segment["negative_ambiguity"]["total"]
        if core_segment["negative_ambiguity"]["total"]
        else 1.0
    )
    core_negative_offtopic_legal_success = (
        core_segment["negative_offtopic_legal"]["clean"]
        / core_segment["negative_offtopic_legal"]["total"]
        if core_segment["negative_offtopic_legal"]["total"]
        else 1.0
    )
    core_negative_offtopic_nonlegal_success = (
        core_segment["negative_offtopic_nonlegal"]["clean"]
        / core_segment["negative_offtopic_nonlegal"]["total"]
        if core_segment["negative_offtopic_nonlegal"]["total"]
        else 1.0
    )
    editorial_context_success = (
        editorial_success / editorial_expected_total if editorial_expected_total else 1.0
    )
    core_editorial_context_success = (
        core_editorial_success / core_editorial_expected_total
        if core_editorial_expected_total
        else 1.0
    )
    non_editorial_clean_success = (
        non_editorial_clean / non_editorial_expected_total if non_editorial_expected_total else 1.0
    )
    core_non_editorial_clean_success = (
        core_non_editorial_clean / core_non_editorial_expected_total
        if core_non_editorial_expected_total
        else 1.0
    )
    mean_expected_coverage = (
        expected_coverage_total / expected_coverage_count if expected_coverage_count else 0.0
    )
    citation_precision = (
        citation_precision_total / citation_precision_count if citation_precision_count else 0.0
    )
    unexpected_citation_rate = (
        unexpected_citation_rate_total / unexpected_citation_rate_count
        if unexpected_citation_rate_count
        else 0.0
    )
    law_contamination_rate = (
        law_contamination_cases / law_contamination_total if law_contamination_total else 0.0
    )
    false_positive_gating_rate = (
        false_positive_gating_count / positive_total if positive_total else 0.0
    )
    coverage_weighted_positive_recall = (
        weighted_positive_recall_numerator / weighted_positive_recall_denominator
        if weighted_positive_recall_denominator
        else 0.0
    )
    recall_at_1 = _safe_div(recall_at_1_hits, positive_total)
    recall_at_3 = _safe_div(recall_at_3_hits, positive_total)
    recall_at_5 = _safe_div(recall_at_5_hits, positive_total)
    mrr_at_5 = _safe_div(reciprocal_rank_sum, mrr_count)
    f1_at_k = _safe_div(
        2.0 * citation_precision * recall_at_k,
        citation_precision + recall_at_k,
    )
    f1_weighted_at_k = _safe_div(
        2.0 * citation_precision * coverage_weighted_positive_recall,
        citation_precision + coverage_weighted_positive_recall,
    )
    source_boundary_mismatch_at_k = _safe_div(
        boundary_wrong_law_count + boundary_wrong_article_same_law_count,
        boundary_items_total,
    )
    boundary_level_a_wrong_law_rate = _safe_div(boundary_wrong_law_count, boundary_items_total)
    boundary_level_b_wrong_article_same_law_rate = _safe_div(
        boundary_wrong_article_same_law_count, boundary_items_total
    )
    boundary_level_c_family_proxy_rate = _safe_div(
        boundary_family_proxy_count, boundary_items_total
    )
    confidence_threshold_crossing_rate = (
        confidence_threshold_crossings / scored_query_count if scored_query_count else 0.0
    )
    ambiguity_threshold_crossing_rate = (
        ambiguity_threshold_crossings / scored_query_count if scored_query_count else 0.0
    )
    calibration_diagnostics: list[dict] = []
    expected_calibration_error = 0.0
    calibration_total = sum(int(bin_item["total"]) for bin_item in calibration_bins)
    for bin_item in calibration_bins:
        total = int(bin_item["total"])
        hits = int(bin_item["hits"])
        observed_precision = (hits / total) if total else 0.0
        predicted_confidence = (float(bin_item["low"]) + float(bin_item["high"])) / 2.0
        bucket_weight = (total / calibration_total) if calibration_total else 0.0
        expected_calibration_error += bucket_weight * abs(observed_precision - predicted_confidence)
        calibration_diagnostics.append(
            {
                "score_range": [
                    round(float(bin_item["low"]), 2),
                    round(float(bin_item["high"]), 2),
                ],
                "total": total,
                "observed_precision": observed_precision,
                "predicted_confidence_midpoint": predicted_confidence,
            }
        )

    metrics = {
        "recall_at_k": recall_at_k,
        "ambiguity_success": ambiguity_success,
        "avg_top_score": avg_top_score,
        "single_article_recall_at_k": single_recall,
        "multi_article_recall_at_k": multi_recall,
        "negative_success": negative_success,
        "core_single_article_recall_at_k": core_single_recall,
        "core_multi_article_recall_at_k": core_multi_recall,
        "core_negative_success": core_negative_success,
        "negative_ambiguity_success": negative_ambiguity_success,
        "negative_offtopic_legal_success": negative_offtopic_legal_success,
        "negative_offtopic_nonlegal_success": negative_offtopic_nonlegal_success,
        "core_negative_ambiguity_success": core_negative_ambiguity_success,
        "core_negative_offtopic_legal_success": core_negative_offtopic_legal_success,
        "core_negative_offtopic_nonlegal_success": core_negative_offtopic_nonlegal_success,
        "mean_expected_coverage": mean_expected_coverage,
        "citation_precision": citation_precision,
        "f1_at_k": f1_at_k,
        "f1_weighted_at_k": f1_weighted_at_k,
        "mrr_at_5": mrr_at_5,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "unexpected_citation_rate": unexpected_citation_rate,
        "law_contamination_rate": law_contamination_rate,
        "source_boundary_mismatch_at_k": source_boundary_mismatch_at_k,
        "boundary_level_a_wrong_law_rate": boundary_level_a_wrong_law_rate,
        "boundary_level_b_wrong_article_same_law_rate": boundary_level_b_wrong_article_same_law_rate,
        "boundary_level_c_family_proxy_rate": boundary_level_c_family_proxy_rate,
        "law_coherence_filtered_count": law_coherence_filtered_count,
        "min_sources_floor_trigger_count": min_sources_floor_trigger_count,
        "confidence_gating_count": confidence_gating_count,
        "ambiguity_gating_count": ambiguity_gating_count,
        "false_positive_gating_rate": false_positive_gating_rate,
        "false_positive_confidence_gating_count": false_positive_confidence_gating_count,
        "false_positive_ambiguity_gating_count": false_positive_ambiguity_gating_count,
        "confidence_threshold_crossing_rate": confidence_threshold_crossing_rate,
        "ambiguity_threshold_crossing_rate": ambiguity_threshold_crossing_rate,
        "coverage_weighted_positive_recall": coverage_weighted_positive_recall,
        "editorial_context_success": editorial_context_success,
        "core_editorial_context_success": core_editorial_context_success,
        "non_editorial_clean_success": non_editorial_clean_success,
        "core_non_editorial_clean_success": core_non_editorial_clean_success,
        "calibration_bins": calibration_diagnostics,
        "expected_calibration_error": expected_calibration_error,
        "routing_uncertain_count": routing_uncertain_count,
        "fallback_stage1_accepted_count": fallback_stage1_accepted_count,
        "fallback_stage1_low_quality_kept_count": fallback_stage1_low_quality_kept_count,
        "fallback_stage2_unfiltered_count": fallback_stage2_unfiltered_count,
        "fallback_stage1_error_count": fallback_stage1_error_count,
    }
    metrics["balanced_score"] = _compute_balanced_score(metrics)
    return metrics


def apply_combo_to_chain(
    chain: LegalRAGChain,
    retrieval_k_initial: int,
    retrieval_k: int,
    confidence: float,
    min_doc: float,
    min_gap: float,
    top_score_ceiling: float,
    routing_fallback_unfiltered: bool,
    reranker_context_enrichment: bool,
    routing_summary_dualpass: bool,
) -> None:
    """Apply sweep parameters to an existing chain instance."""
    chain.settings.retrieval_k_initial = retrieval_k_initial
    chain.settings.retrieval_k = retrieval_k
    chain.settings.reranker_confidence_threshold = confidence
    chain.settings.reranker_min_doc_score = min_doc
    chain.settings.reranker_ambiguity_min_gap = min_gap
    chain.settings.reranker_ambiguity_top_score_ceiling = top_score_ceiling
    chain.settings.law_routing_fallback_unfiltered = routing_fallback_unfiltered
    chain.settings.reranker_context_enrichment_enabled = reranker_context_enrichment
    chain.settings.law_routing_summary_dualpass_enabled = routing_summary_dualpass


def main() -> None:
    # Keep local sweeps from consuming LangSmith quota.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    if load_dotenv:
        load_dotenv(ROOT_DIR / ".env")
    questions_path = ROOT_DIR / "eval" / "questions.jsonl"
    questions = load_questions(questions_path)
    sample_size_raw = os.getenv("SWEEP_SAMPLE_SIZE")
    if sample_size_raw:
        sample_size = int(sample_size_raw)
        if sample_size > 0:
            questions = questions[:sample_size]
            logger.info("Using sample size from SWEEP_SAMPLE_SIZE=%s", sample_size)
    settings = get_settings()
    profile_name = os.getenv("TRUST_PROFILE", settings.trust_profile_name)
    resolved_profile = apply_trust_profile(settings, profile_name)
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = (
        os.getenv("LOVLI_RUN_ID")
        or f"sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    questions_sha256 = _sha256_file(questions_path)
    git_commit = _safe_git_commit()
    logger.info(
        "Using trust profile: %s (version=%s)",
        resolved_profile,
        settings.trust_profile_version,
    )
    logger.info(
        "Run metadata: run_id=%s git_commit=%s questions_sha256=%s",
        run_id,
        git_commit,
        questions_sha256[:12],
    )
    profile_defaults = {
        "retrieval_k_initial": settings.retrieval_k_initial,
        "retrieval_k": settings.retrieval_k,
        "reranker_confidence_threshold": settings.reranker_confidence_threshold,
        "reranker_min_doc_score": settings.reranker_min_doc_score,
        "reranker_ambiguity_min_gap": settings.reranker_ambiguity_min_gap,
        "reranker_ambiguity_top_score_ceiling": settings.reranker_ambiguity_top_score_ceiling,
        "law_routing_fallback_unfiltered": settings.law_routing_fallback_unfiltered,
        "reranker_context_enrichment_enabled": settings.reranker_context_enrichment_enabled,
        "law_routing_summary_dualpass_enabled": settings.law_routing_summary_dualpass_enabled,
        "law_coherence_dominant_concentration_threshold": settings.law_coherence_dominant_concentration_threshold,
        "law_routing_stage1_min_docs": settings.law_routing_stage1_min_docs,
        "law_routing_stage1_min_top_score": settings.law_routing_stage1_min_top_score,
        "law_rank_fusion_enabled": settings.law_rank_fusion_enabled,
    }

    quick_grid = _env_flag("SWEEP_QUICK_GRID", default=False)
    debug_grid = _env_flag("SWEEP_DEBUG_MODE", default=False)

    if quick_grid and debug_grid:
        logger.warning("Both SWEEP_QUICK_GRID and SWEEP_DEBUG_MODE are set. Using quick_grid.")
    retrieval_k_initial_values: list[int] = []
    retrieval_k_values: list[int] = []
    confidence_values: list[float] = []
    min_doc_values: list[float] = []
    min_gap_values: list[float] = []
    top_score_ceiling_values: list[float] = []
    routing_fallback_unfiltered_values: list[bool] = []
    reranker_context_enrichment_values: list[bool] = []
    routing_summary_dualpass_values: list[bool] = []
    if quick_grid:
        retrieval_k_initial_values = [int(settings.retrieval_k_initial)]
        retrieval_k_values = [int(settings.retrieval_k)]
        confidence_values = [float(settings.reranker_confidence_threshold)]
        min_doc_values = [float(settings.reranker_min_doc_score)]
        min_gap_values = [float(settings.reranker_ambiguity_min_gap)]
        top_score_ceiling_values = [float(settings.reranker_ambiguity_top_score_ceiling)]
        routing_fallback_unfiltered_values = [bool(settings.law_routing_fallback_unfiltered)]
        reranker_context_enrichment_values = [bool(settings.reranker_context_enrichment_enabled)]
        routing_summary_dualpass_values = [bool(settings.law_routing_summary_dualpass_enabled)]
        logger.info("SWEEP_QUICK_GRID=true: using profile-default values only (single combo)")
    elif debug_grid:
        retrieval_k_initial_values = [int(settings.retrieval_k_initial)]
        retrieval_k_values = [3, 4, 5]
        confidence_values = [0.25, 0.30, 0.35, 0.40]
        min_doc_values = [0.25, 0.30, 0.35, 0.40]
        min_gap_values = [float(settings.reranker_ambiguity_min_gap)]
        top_score_ceiling_values = [float(settings.reranker_ambiguity_top_score_ceiling)]
        routing_fallback_unfiltered_values = [bool(settings.law_routing_fallback_unfiltered)]
        reranker_context_enrichment_values = [bool(settings.reranker_context_enrichment_enabled)]
        routing_summary_dualpass_values = [bool(settings.law_routing_summary_dualpass_enabled)]
        logger.info(
            "SWEEP_DEBUG_MODE=true: using focused debug grid (%dx%dx%d=%d combos)",
            len(retrieval_k_values),
            len(confidence_values),
            len(min_doc_values),
            len(retrieval_k_values) * len(confidence_values) * len(min_doc_values),
        )
    else:
        retrieval_k_initial_values = _with_default_value(
            [15, 20], int(settings.retrieval_k_initial)
        )
        retrieval_k_values = _with_default_value([3, 5], int(settings.retrieval_k))
        confidence_values = _with_default_value(
            [0.30, 0.35, 0.45, 0.55],
            float(settings.reranker_confidence_threshold),
        )
        min_doc_values = _with_default_value(
            [0.25, 0.35, 0.45, 0.55],
            float(settings.reranker_min_doc_score),
        )
        min_gap_values = _with_default_value(
            [0.05, 0.10], float(settings.reranker_ambiguity_min_gap)
        )
        top_score_ceiling_values = _with_default_value(
            [0.60, 0.70],
            float(settings.reranker_ambiguity_top_score_ceiling),
        )
        routing_fallback_unfiltered_values = [True, False]
    if not quick_grid and not debug_grid:
        if bool(settings.law_routing_fallback_unfiltered) not in routing_fallback_unfiltered_values:
            routing_fallback_unfiltered_values.append(
                bool(settings.law_routing_fallback_unfiltered)
            )
        routing_fallback_unfiltered_values = sorted(
            set(routing_fallback_unfiltered_values), reverse=True
        )
        reranker_context_enrichment_values = [True, False]
        if (
            bool(settings.reranker_context_enrichment_enabled)
            not in reranker_context_enrichment_values
        ):
            reranker_context_enrichment_values.append(
                bool(settings.reranker_context_enrichment_enabled)
            )
        reranker_context_enrichment_values = sorted(
            set(reranker_context_enrichment_values), reverse=True
        )
        routing_summary_dualpass_values = [False, True]
        if (
            bool(settings.law_routing_summary_dualpass_enabled)
            not in routing_summary_dualpass_values
        ):
            routing_summary_dualpass_values.append(
                bool(settings.law_routing_summary_dualpass_enabled)
            )
        routing_summary_dualpass_values = sorted(set(routing_summary_dualpass_values), reverse=True)

    logger.info("Loaded %s questions", len(questions))
    core_count = sum(1 for row in questions if row.get("id") in CORE_QUESTION_IDS)
    logger.info("Frozen core subset size: %s", core_count)
    logger.info(
        "Sweep grid values: k_init=%s k=%s conf=%s min_doc=%s min_gap=%s top_ceiling=%s "
        "route_unfiltered=%s reranker_ctx=%s dualpass=%s",
        retrieval_k_initial_values,
        retrieval_k_values,
        confidence_values,
        min_doc_values,
        min_gap_values,
        top_score_ceiling_values,
        routing_fallback_unfiltered_values,
        reranker_context_enrichment_values,
        routing_summary_dualpass_values,
    )
    logger.info("Starting retrieval sweep...")
    chain = LegalRAGChain(settings=settings)
    # After the main chain loads all models from HuggingFace, switch to offline mode so
    # parallel precompute workers (which create fresh LegalRAGChain instances) load from
    # the local disk cache rather than hitting the HF Hub API and triggering 429 rate limits.
    if os.environ.get("HF_HUB_OFFLINE") != "0":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.info("HF Hub offline mode enabled for parallel workers")
    skip_index_scan = _env_flag("SWEEP_SKIP_INDEX_SCAN", default=True)
    validate_questions(questions, chain, skip_index_scan=skip_index_scan)
    max_k_initial = max(retrieval_k_initial_values)
    cache_dir_raw = os.getenv("SWEEP_CACHE_DIR")
    cache_dir = Path(cache_dir_raw) if cache_dir_raw and cache_dir_raw.strip() else None
    cached_candidates_by_mode: dict[tuple[bool, bool, bool], list[dict]] = {}
    qdrant_collection = settings.qdrant_collection_name
    for fallback_unfiltered, reranker_ctx_enabled, dualpass_enabled in itertools.product(
        routing_fallback_unfiltered_values,
        reranker_context_enrichment_values,
        routing_summary_dualpass_values,
    ):
        chain.settings.law_routing_fallback_unfiltered = fallback_unfiltered
        chain.settings.reranker_context_enrichment_enabled = reranker_ctx_enabled
        chain.settings.law_routing_summary_dualpass_enabled = dualpass_enabled
        cache_key_tuple = (fallback_unfiltered, reranker_ctx_enabled, dualpass_enabled)
        key_hash = _precompute_cache_key(
            questions_sha256,
            git_commit,
            qdrant_collection,
            max_k_initial,
            fallback_unfiltered,
            reranker_ctx_enabled,
            dualpass_enabled,
        )
        cache_path = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"sweep_precompute_{key_hash}.json"
        loaded = False
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if (
                    data.get("questions_sha256") == questions_sha256
                    and data.get("git_commit") == git_commit
                    and data.get("qdrant_collection") == qdrant_collection
                    and data.get("max_k_initial") == max_k_initial
                ):
                    cached_candidates_by_mode[cache_key_tuple] = data["candidates"]
                    loaded = True
                    logger.info(
                        "Loaded precompute cache %s (route_unfiltered=%s reranker_ctx=%s dualpass=%s)",
                        key_hash[:12],
                        fallback_unfiltered,
                        reranker_ctx_enabled,
                        dualpass_enabled,
                    )
            except Exception as e:
                logger.warning("Cache load failed for %s: %s", cache_path, e)
        if not loaded:
            parallel_precompute = _env_flag("SWEEP_PARALLEL_PRECOMPUTE", default=False)
            max_workers = int(os.getenv("SWEEP_PARALLEL_WORKERS", "4"))
            logger.info(
                "Precomputing candidates with k=%s (route_unfiltered=%s reranker_ctx=%s dualpass=%s)...",
                max_k_initial,
                fallback_unfiltered,
                reranker_ctx_enabled,
                dualpass_enabled,
            )
            cached_candidates_by_mode[cache_key_tuple] = precompute_candidates(
                chain,
                questions,
                max_k_initial=max_k_initial,
                parallel=parallel_precompute,
                max_workers=max_workers,
            )
            if cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "questions_sha256": questions_sha256,
                                "git_commit": git_commit,
                                "qdrant_collection": qdrant_collection,
                                "max_k_initial": max_k_initial,
                                "candidates": cached_candidates_by_mode[cache_key_tuple],
                            },
                            f,
                            ensure_ascii=False,
                        )
                    logger.info("Saved precompute cache %s", key_hash[:12])
                except Exception as e:
                    logger.warning("Cache save failed: %s", e)

    enable_checkpoint = _env_flag("SWEEP_CHECKPOINT", default=True)
    checkpoint_interval = int(os.getenv("SWEEP_CHECKPOINT_INTERVAL", "10"))
    checkpoint_path = ROOT_DIR / "eval" / f"sweep_checkpoint_{run_id}.json"
    rows: list[dict] = []
    if enable_checkpoint and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            if checkpoint_data.get("run_id") == run_id:
                rows = checkpoint_data.get("rows", [])
                logger.info("Resumed from checkpoint: %d rows already completed", len(rows))
        except Exception as e:
            logger.warning("Checkpoint load failed: %s", e)

    def save_checkpoint():
        if enable_checkpoint:
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({"run_id": run_id, "rows": rows}, f, ensure_ascii=False)
            except Exception as e:
                logger.warning("Checkpoint save failed: %s", e)

    total_combos = (
        len(retrieval_k_initial_values)
        * len(retrieval_k_values)
        * len(confidence_values)
        * len(min_doc_values)
        * len(min_gap_values)
        * len(top_score_ceiling_values)
        * len(routing_fallback_unfiltered_values)
        * len(reranker_context_enrichment_values)
        * len(routing_summary_dualpass_values)
    )
    logger.info("Total combos to evaluate: %d", total_combos)

    if enable_checkpoint and rows:
        existing_combo_keys = {
            (
                r.get("retrieval_k_initial"),
                r.get("retrieval_k"),
                r.get("reranker_confidence_threshold"),
                r.get("reranker_min_doc_score"),
                r.get("reranker_ambiguity_min_gap"),
                r.get("reranker_ambiguity_top_score_ceiling"),
                r.get("law_routing_fallback_unfiltered"),
                r.get("reranker_context_enrichment_enabled"),
                r.get("law_routing_summary_dualpass_enabled"),
            )
            for r in rows
        }
        logger.info("Found %d already completed combos in checkpoint", len(existing_combo_keys))
    else:
        existing_combo_keys = set()

    for (
        retrieval_k_initial,
        retrieval_k,
        confidence,
        min_doc,
        min_gap,
        top_score_ceiling,
        routing_fallback_unfiltered,
        reranker_context_enrichment,
        routing_summary_dualpass,
    ) in itertools.product(
        retrieval_k_initial_values,
        retrieval_k_values,
        confidence_values,
        min_doc_values,
        min_gap_values,
        top_score_ceiling_values,
        routing_fallback_unfiltered_values,
        reranker_context_enrichment_values,
        routing_summary_dualpass_values,
    ):
        combo_key = (
            retrieval_k_initial,
            retrieval_k,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            routing_fallback_unfiltered,
            reranker_context_enrichment,
            routing_summary_dualpass,
        )
        if combo_key in existing_combo_keys:
            continue
        apply_combo_to_chain(
            chain,
            retrieval_k_initial,
            retrieval_k,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            routing_fallback_unfiltered,
            reranker_context_enrichment,
            routing_summary_dualpass,
        )
        metrics = evaluate_combo(
            chain,
            cached_candidates_by_mode[
                (routing_fallback_unfiltered, reranker_context_enrichment, routing_summary_dualpass)
            ],
            retrieval_k_initial=retrieval_k_initial,
            confidence_threshold=confidence,
            min_doc_score=min_doc,
            ambiguity_min_gap=min_gap,
            top_score_ceiling=top_score_ceiling,
        )
        row = {
            "run_id": run_id,
            "run_started_at": run_started_at,
            "git_commit": git_commit,
            "questions_sha256": questions_sha256,
            "question_count": len(questions),
            "trust_profile_name": settings.trust_profile_name,
            "trust_profile_version": settings.trust_profile_version,
            "sweep_source_profile_name": resolved_profile,
            "retrieval_k_initial": retrieval_k_initial,
            "retrieval_k": retrieval_k,
            "reranker_confidence_threshold": confidence,
            "reranker_min_doc_score": min_doc,
            "reranker_ambiguity_min_gap": min_gap,
            "reranker_ambiguity_top_score_ceiling": top_score_ceiling,
            "law_routing_fallback_unfiltered": routing_fallback_unfiltered,
            "reranker_context_enrichment_enabled": reranker_context_enrichment,
            "law_routing_summary_dualpass_enabled": routing_summary_dualpass,
            "law_coherence_dominant_concentration_threshold": (
                chain.settings.law_coherence_dominant_concentration_threshold
            ),
            "law_routing_stage1_min_docs": chain.settings.law_routing_stage1_min_docs,
            "law_routing_stage1_min_top_score": chain.settings.law_routing_stage1_min_top_score,
            "law_rank_fusion_enabled": chain.settings.law_rank_fusion_enabled,
            **metrics,
        }
        row["is_profile_default_row"] = _is_profile_default_row(row, profile_defaults)
        rows.append(row)
        logger.info(
            "k_init=%s k=%s conf=%.2f min_doc=%.2f min_gap=%.2f top_ceiling=%.2f "
            "route_unfiltered=%s reranker_ctx=%s dualpass=%s -> "
            "balanced=%.3f recall=%.3f coverage=%.3f precision=%.3f unexpected=%.3f "
            "law_contam=%.3f coherence_dropped=%s "
            "fp_gate=%.3f cw_recall=%.3f mrr5=%.3f r@1=%.3f r@3=%.3f r@5=%.3f "
            "boundary_mismatch=%.3f floor_triggers=%s conf_gate=%s amb_gate=%s "
            "uncertain=%s stage2_unfiltered=%s "
            "single=%.3f multi=%.3f neg=%.3f neg_legal=%.3f neg_nonlegal=%.3f "
            "editorial=%.3f non_editorial_clean=%.3f avg_top=%.3f",
            retrieval_k_initial,
            retrieval_k,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            routing_fallback_unfiltered,
            reranker_context_enrichment,
            routing_summary_dualpass,
            metrics["balanced_score"],
            metrics["recall_at_k"],
            metrics["mean_expected_coverage"],
            metrics["citation_precision"],
            metrics["unexpected_citation_rate"],
            metrics["law_contamination_rate"],
            metrics["law_coherence_filtered_count"],
            metrics["false_positive_gating_rate"],
            metrics["coverage_weighted_positive_recall"],
            metrics["mrr_at_5"],
            metrics["recall_at_1"],
            metrics["recall_at_3"],
            metrics["recall_at_5"],
            metrics["source_boundary_mismatch_at_k"],
            metrics["min_sources_floor_trigger_count"],
            metrics["confidence_gating_count"],
            metrics["ambiguity_gating_count"],
            metrics["routing_uncertain_count"],
            metrics["fallback_stage2_unfiltered_count"],
            metrics["single_article_recall_at_k"],
            metrics["multi_article_recall_at_k"],
            metrics["negative_success"],
            metrics["negative_offtopic_legal_success"],
            metrics["negative_offtopic_nonlegal_success"],
            metrics["editorial_context_success"],
            metrics["non_editorial_clean_success"],
            metrics["avg_top_score"],
        )
        if enable_checkpoint and len(rows) % checkpoint_interval == 0:
            save_checkpoint()
            logger.info("Checkpoint saved: %d/%d combos evaluated", len(rows), total_combos)

    # Sort by balanced objective first, then tie-breakers.
    rows.sort(
        key=lambda r: (
            r["balanced_score"],
            r["mrr_at_5"],
            r["recall_at_1"],
            r["coverage_weighted_positive_recall"],
            r["recall_at_k"],
            r["citation_precision"],
            -r["source_boundary_mismatch_at_k"],
            -r["law_contamination_rate"],
            -r["false_positive_gating_rate"],
            -r["unexpected_citation_rate"],
            r["negative_success"],
        ),
        reverse=True,
    )
    default_row_count = sum(1 for row in rows if bool(row.get("is_profile_default_row")))
    if default_row_count != 1:
        raise ValueError(
            f"Expected exactly one profile-default row, found {default_row_count}. "
            "Ensure sweep grid includes active trust profile defaults."
        )
    default_row = next(row for row in rows if bool(row.get("is_profile_default_row")))
    improvement_delta = float(os.getenv("SWEEP_PROMOTION_MIN_IMPROVEMENT", "0.01"))
    precision_tolerance = float(os.getenv("SWEEP_PROMOTION_PRECISION_TOLERANCE", "0.005"))
    negative_tolerance = float(os.getenv("SWEEP_PROMOTION_NEGATIVE_TOLERANCE", "0.010"))
    baseline_snapshot = {
        "source_boundary_mismatch_at_k": float(
            default_row.get("source_boundary_mismatch_at_k", 0.0)
        ),
        "mrr_at_5": float(default_row.get("mrr_at_5", 0.0)),
        "recall_at_5": float(default_row.get("recall_at_5", 0.0)),
        "citation_precision": float(default_row.get("citation_precision", 0.0)),
        "negative_success": float(default_row.get("negative_success", 0.0)),
    }
    for row in rows:
        reasons: list[str] = []
        boundary_gain = baseline_snapshot["source_boundary_mismatch_at_k"] - float(
            row.get("source_boundary_mismatch_at_k", 0.0)
        )
        mrr_gain = float(row.get("mrr_at_5", 0.0)) - baseline_snapshot["mrr_at_5"]
        recall5_gain = float(row.get("recall_at_5", 0.0)) - baseline_snapshot["recall_at_5"]
        citation_delta = (
            float(row.get("citation_precision", 0.0)) - baseline_snapshot["citation_precision"]
        )
        negative_delta = (
            float(row.get("negative_success", 0.0)) - baseline_snapshot["negative_success"]
        )

        if boundary_gain < improvement_delta:
            reasons.append("insufficient_boundary_mismatch_gain")
        if mrr_gain < improvement_delta:
            reasons.append("insufficient_mrr_gain")
        if recall5_gain < improvement_delta:
            reasons.append("insufficient_recall5_gain")
        if citation_delta < (-precision_tolerance):
            reasons.append("citation_precision_regression")
        if negative_delta < (-negative_tolerance):
            reasons.append("negative_success_regression")

        row["promotion_gate_pass"] = len(reasons) == 0
        row["promotion_gate_reasons"] = reasons
        row["promotion_gate_baseline"] = baseline_snapshot

    out_path = ROOT_DIR / "eval" / "retrieval_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    summary_path = ROOT_DIR / "eval" / "retrieval_sweep_summary.json"

    pair_index: dict[tuple, dict[bool, dict]] = {}
    for row in rows:
        key = (
            row.get("retrieval_k_initial"),
            row.get("retrieval_k"),
            row.get("reranker_confidence_threshold"),
            row.get("reranker_min_doc_score"),
            row.get("reranker_ambiguity_min_gap"),
            row.get("reranker_ambiguity_top_score_ceiling"),
            row.get("reranker_context_enrichment_enabled"),
            row.get("law_routing_summary_dualpass_enabled"),
        )
        pair_index.setdefault(key, {})[bool(row.get("law_routing_fallback_unfiltered"))] = row
    compared_pairs = 0
    identical_pairs = 0
    divergence_counts = {
        "recall_at_k": 0,
        "recall_at_1": 0,
        "mrr_at_5": 0,
        "citation_precision": 0,
        "unexpected_citation_rate": 0,
        "false_positive_gating_rate": 0,
        "law_contamination_rate": 0,
        "source_boundary_mismatch_at_k": 0,
        "balanced_score": 0,
    }
    delta_sums = {key: 0.0 for key in divergence_counts}
    delta_max = {key: 0.0 for key in divergence_counts}
    for pair in pair_index.values():
        if True not in pair or False not in pair:
            continue
        compared_pairs += 1
        left = pair[True]
        right = pair[False]
        for metric in divergence_counts:
            left_value = float(left.get(metric, 0.0))
            right_value = float(right.get(metric, 0.0))
            diff = abs(left_value - right_value)
            if diff > 1e-9:
                divergence_counts[metric] += 1
            delta_sums[metric] += diff
            delta_max[metric] = max(delta_max[metric], diff)
        if (
            float(left.get("recall_at_k", 0.0)) == float(right.get("recall_at_k", 0.0))
            and float(left.get("citation_precision", 0.0))
            == float(right.get("citation_precision", 0.0))
            and float(left.get("unexpected_citation_rate", 0.0))
            == float(right.get("unexpected_citation_rate", 0.0))
        ):
            identical_pairs += 1
    logger.info(
        "Parity debug: compared_fallback_pairs=%s identical_metric_pairs=%s",
        compared_pairs,
        identical_pairs,
    )
    if compared_pairs > 0:
        logger.info(
            "Parity divergence counts: %s",
            {key: f"{count}/{compared_pairs}" for key, count in divergence_counts.items()},
        )
        logger.info(
            "Parity absolute delta summary: %s",
            {
                key: {
                    "avg_delta": (delta_sums[key] / compared_pairs),
                    "max_delta": delta_max[key],
                }
                for key in divergence_counts
            },
        )
    if compared_pairs > 0 and identical_pairs == compared_pairs:
        logger.warning(
            "Fallback toggle produced identical metrics across all compared pairs. "
            "Inspect routing/fallback candidate split diagnostics."
        )

    logger.info("Saved results: %s", out_path)
    promotion_pass_count = sum(1 for row in rows if bool(row.get("promotion_gate_pass")))
    logger.info(
        "Promotion gate summary: pass=%s/%s (delta=%.3f precision_tol=%.3f negative_tol=%.3f)",
        promotion_pass_count,
        len(rows),
        improvement_delta,
        precision_tolerance,
        negative_tolerance,
    )
    logger.info("Top 5 configurations:")
    for row in rows[:5]:
        logger.info(row)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "git_commit": git_commit,
                "questions_sha256": questions_sha256,
                "question_count": len(questions),
                "rows_count": len(rows),
                "promotion_gate_pass_count": promotion_pass_count,
                "promotion_gate_total": len(rows),
                "promotion_gate_thresholds": {
                    "improvement_delta": improvement_delta,
                    "precision_tolerance": precision_tolerance,
                    "negative_tolerance": negative_tolerance,
                },
                "top_configuration": rows[0] if rows else {},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Saved ablation summary: %s", summary_path)

    if enable_checkpoint and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.info("Checkpoint file cleaned up after successful completion")
        except Exception as e:
            logger.warning("Checkpoint cleanup failed: %s", e)


if __name__ == "__main__":
    main()
