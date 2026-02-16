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
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain, _infer_doc_type  # noqa: E402
from lovli.config import get_settings  # noqa: E402
from lovli.eval_utils import infer_negative_type, validate_questions  # noqa: E402
from lovli.retrieval_shared import (  # noqa: E402
    apply_uncertainty_law_cap,
    build_law_aware_rank_fusion,
    build_law_cross_reference_affinity,
    build_law_coherence_decision,
    matches_expected_source,
    normalize_sigmoid_scores,
)
from lovli.trust_profiles import apply_trust_profile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Frozen core subset for stable regressions across tuning runs.
CORE_QUESTION_IDS = {
    "q001", "q002", "q003", "q004", "q005", "q006", "q007", "q008", "q009", "q010",
    "q011", "q012", "q013", "q014", "q015", "q016", "q017", "q018", "q019", "q020",
    "q021", "q022", "q031", "q036", "q037", "q038", "q039", "q040", "q041", "q042", "q045",
    "q046", "q047", "q048", "q049", "q050",
}


def matches_expected(cited_id: str, expected_set: set[str]) -> bool:
    """Prefix-match cited article IDs against expected IDs."""
    return any(cited_id.startswith(exp) for exp in expected_set)


def load_questions(path: Path) -> list[dict]:
    questions: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


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
        (0.30 * metrics.get("recall_at_k", 0.0))
        + (0.20 * metrics.get("mean_expected_coverage", 0.0))
        + (0.15 * metrics.get("citation_precision", 0.0))
    )
    abstention_component = (
        (0.20 * metrics.get("negative_success", 0.0))
        + (0.10 * metrics.get("negative_offtopic_legal_success", 0.0))
        + (0.05 * metrics.get("negative_offtopic_nonlegal_success", 0.0))
    )
    penalties = (
        (0.08 * metrics.get("law_contamination_rate", 0.0))
        + (0.07 * metrics.get("unexpected_citation_rate", 0.0))
        + (0.10 * metrics.get("false_positive_gating_rate", 0.0))
    )
    return accuracy_component + abstention_component - penalties


def _is_profile_default_row(row: dict, profile_defaults: dict) -> bool:
    """Return True when a sweep row exactly matches active profile defaults."""
    for key, expected in profile_defaults.items():
        if row.get(key) != expected:
            return False
    return True


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


def precompute_candidates(
    chain: LegalRAGChain,
    questions: list[dict],
    max_k_initial: int,
) -> list[dict]:
    """Run retrieval/reranker once per question and cache candidates for offline sweeping."""
    chain.retriever = chain.vectorstore.as_retriever(search_kwargs={"k": max_k_initial})
    cached: list[dict] = []
    for row in questions:
        query = row["question"]
        routed_law_ids = chain._route_law_ids(query)
        route_diag_before = dict(getattr(chain, "_last_routing_diagnostics", {}) or {})
        docs = chain._invoke_retriever(query, routed_law_ids=routed_law_ids)
        route_diag_after = dict(getattr(chain, "_last_routing_diagnostics", route_diag_before) or route_diag_before)

        # Deduplicate like retrieve()
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
            pairs = [
                [query, doc.page_content if hasattr(doc, "page_content") else str(doc)]
                for doc in docs
            ]
            raw_scores = chain.reranker.predict(pairs) if chain.reranker else [1.0] * len(docs)
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

        cached.append(
            {
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
                "candidates": candidates,
            }
        )
    return cached


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
        {"low": i / calibration_bin_count, "high": (i + 1) / calibration_bin_count, "total": 0, "hits": 0}
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
    top_scores: list[float] = []
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
        provision_kept, coherence_dropped, coherence_decision = _apply_law_coherence_filter_candidates(
            provision_kept,
            chain.settings,
            law_ref_to_id=getattr(chain, "_law_ref_to_id", {}) or {},
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
                    (len(cited_ids) - matched_citations) / len(cited_ids)
                    if cited_ids
                    else 0.0
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
        segment["negative_offtopic_nonlegal"]["clean"] / segment["negative_offtopic_nonlegal"]["total"]
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
        core_segment["negative_offtopic_legal"]["clean"] / core_segment["negative_offtopic_legal"]["total"]
        if core_segment["negative_offtopic_legal"]["total"]
        else 1.0
    )
    core_negative_offtopic_nonlegal_success = (
        core_segment["negative_offtopic_nonlegal"]["clean"] / core_segment["negative_offtopic_nonlegal"]["total"]
        if core_segment["negative_offtopic_nonlegal"]["total"]
        else 1.0
    )
    editorial_context_success = (
        editorial_success / editorial_expected_total
        if editorial_expected_total
        else 1.0
    )
    core_editorial_context_success = (
        core_editorial_success / core_editorial_expected_total
        if core_editorial_expected_total
        else 1.0
    )
    non_editorial_clean_success = (
        non_editorial_clean / non_editorial_expected_total
        if non_editorial_expected_total
        else 1.0
    )
    core_non_editorial_clean_success = (
        core_non_editorial_clean / core_non_editorial_expected_total
        if core_non_editorial_expected_total
        else 1.0
    )
    mean_expected_coverage = (
        expected_coverage_total / expected_coverage_count
        if expected_coverage_count
        else 0.0
    )
    citation_precision = (
        citation_precision_total / citation_precision_count
        if citation_precision_count
        else 0.0
    )
    unexpected_citation_rate = (
        unexpected_citation_rate_total / unexpected_citation_rate_count
        if unexpected_citation_rate_count
        else 0.0
    )
    law_contamination_rate = (
        law_contamination_cases / law_contamination_total
        if law_contamination_total
        else 0.0
    )
    false_positive_gating_rate = (
        false_positive_gating_count / positive_total if positive_total else 0.0
    )
    coverage_weighted_positive_recall = (
        weighted_positive_recall_numerator / weighted_positive_recall_denominator
        if weighted_positive_recall_denominator
        else 0.0
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
                "score_range": [round(float(bin_item["low"]), 2), round(float(bin_item["high"]), 2)],
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
        "unexpected_citation_rate": unexpected_citation_rate,
        "law_contamination_rate": law_contamination_rate,
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
) -> None:
    """Apply sweep parameters to an existing chain instance."""
    chain.settings.retrieval_k_initial = retrieval_k_initial
    chain.settings.retrieval_k = retrieval_k
    chain.settings.reranker_confidence_threshold = confidence
    chain.settings.reranker_min_doc_score = min_doc
    chain.settings.reranker_ambiguity_min_gap = min_gap
    chain.settings.reranker_ambiguity_top_score_ceiling = top_score_ceiling
    chain.settings.law_routing_fallback_unfiltered = routing_fallback_unfiltered


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
    logger.info(
        "Using trust profile: %s (version=%s)",
        resolved_profile,
        settings.trust_profile_version,
    )
    profile_defaults = {
        "retrieval_k_initial": settings.retrieval_k_initial,
        "retrieval_k": settings.retrieval_k,
        "reranker_confidence_threshold": settings.reranker_confidence_threshold,
        "reranker_min_doc_score": settings.reranker_min_doc_score,
        "reranker_ambiguity_min_gap": settings.reranker_ambiguity_min_gap,
        "reranker_ambiguity_top_score_ceiling": settings.reranker_ambiguity_top_score_ceiling,
        "law_routing_fallback_unfiltered": settings.law_routing_fallback_unfiltered,
        "law_coherence_dominant_concentration_threshold": settings.law_coherence_dominant_concentration_threshold,
        "law_routing_stage1_min_docs": settings.law_routing_stage1_min_docs,
        "law_routing_stage1_min_top_score": settings.law_routing_stage1_min_top_score,
        "law_rank_fusion_enabled": settings.law_rank_fusion_enabled,
    }

    retrieval_k_initial_values = [15, 20]
    retrieval_k_values = [3, 5]
    confidence_values = [0.30, 0.35, 0.45, 0.55]
    min_doc_values = [0.25, 0.35, 0.45, 0.55]
    min_gap_values = [0.05, 0.10]
    top_score_ceiling_values = [0.60, 0.70]
    routing_fallback_unfiltered_values = [True, False]

    logger.info("Loaded %s questions", len(questions))
    core_count = sum(1 for row in questions if row.get("id") in CORE_QUESTION_IDS)
    logger.info("Frozen core subset size: %s", core_count)
    logger.info("Starting retrieval sweep...")
    chain = LegalRAGChain(settings=settings)
    skip_index_scan = _env_flag("SWEEP_SKIP_INDEX_SCAN", default=True)
    validate_questions(questions, chain, skip_index_scan=skip_index_scan)
    max_k_initial = max(retrieval_k_initial_values)
    cached_candidates_by_fallback: dict[bool, list[dict]] = {}
    for fallback_unfiltered in routing_fallback_unfiltered_values:
        chain.settings.law_routing_fallback_unfiltered = fallback_unfiltered
        logger.info(
            "Precomputing candidates with k=%s (routing_fallback_unfiltered=%s)...",
            max_k_initial,
            fallback_unfiltered,
        )
        cached_candidates_by_fallback[fallback_unfiltered] = precompute_candidates(
            chain, questions, max_k_initial=max_k_initial
        )

    rows: list[dict] = []
    for (
        retrieval_k_initial,
        retrieval_k,
        confidence,
        min_doc,
        min_gap,
        top_score_ceiling,
        routing_fallback_unfiltered,
    ) in itertools.product(
        retrieval_k_initial_values,
        retrieval_k_values,
        confidence_values,
        min_doc_values,
        min_gap_values,
        top_score_ceiling_values,
        routing_fallback_unfiltered_values,
    ):
        apply_combo_to_chain(
            chain,
            retrieval_k_initial,
            retrieval_k,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            routing_fallback_unfiltered,
        )
        metrics = evaluate_combo(
            chain,
            cached_candidates_by_fallback[routing_fallback_unfiltered],
            retrieval_k_initial=retrieval_k_initial,
            confidence_threshold=confidence,
            min_doc_score=min_doc,
            ambiguity_min_gap=min_gap,
            top_score_ceiling=top_score_ceiling,
        )
        row = {
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
            "k_init=%s k=%s conf=%.2f min_doc=%.2f min_gap=%.2f top_ceiling=%.2f route_unfiltered=%s -> "
            "balanced=%.3f recall=%.3f coverage=%.3f precision=%.3f unexpected=%.3f "
            "law_contam=%.3f coherence_dropped=%s "
            "fp_gate=%.3f cw_recall=%.3f floor_triggers=%s conf_gate=%s amb_gate=%s "
            "single=%.3f multi=%.3f neg=%.3f neg_legal=%.3f neg_nonlegal=%.3f "
            "editorial=%.3f non_editorial_clean=%.3f avg_top=%.3f",
            retrieval_k_initial,
            retrieval_k,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            routing_fallback_unfiltered,
            metrics["balanced_score"],
            metrics["recall_at_k"],
            metrics["mean_expected_coverage"],
            metrics["citation_precision"],
            metrics["unexpected_citation_rate"],
            metrics["law_contamination_rate"],
            metrics["law_coherence_filtered_count"],
            metrics["false_positive_gating_rate"],
            metrics["coverage_weighted_positive_recall"],
            metrics["min_sources_floor_trigger_count"],
            metrics["confidence_gating_count"],
            metrics["ambiguity_gating_count"],
            metrics["single_article_recall_at_k"],
            metrics["multi_article_recall_at_k"],
            metrics["negative_success"],
            metrics["negative_offtopic_legal_success"],
            metrics["negative_offtopic_nonlegal_success"],
            metrics["editorial_context_success"],
            metrics["non_editorial_clean_success"],
            metrics["avg_top_score"],
        )

    # Sort by balanced objective first, then tie-breakers.
    rows.sort(
        key=lambda r: (
            r["balanced_score"],
            r["coverage_weighted_positive_recall"],
            r["recall_at_k"],
            r["citation_precision"],
            -r["law_contamination_rate"],
            -r["false_positive_gating_rate"],
            -r["unexpected_citation_rate"],
            r["negative_success"],
        ),
        reverse=True,
    )
    out_path = ROOT_DIR / "eval" / "retrieval_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("Saved results: %s", out_path)
    logger.info("Top 5 configurations:")
    for row in rows[:5]:
        logger.info(row)


if __name__ == "__main__":
    main()
