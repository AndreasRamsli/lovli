#!/usr/bin/env python3
"""
Run a lightweight retrieval quality sweep over threshold combinations.

This evaluates retrieval/reranking behavior only (no answer generation), so
you can quickly pick robust defaults before full LangSmith runs.
"""

import itertools
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from collections import Counter

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in some environments
    load_dotenv = None

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain  # noqa: E402
from lovli.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Frozen core subset for stable regressions across tuning runs.
CORE_QUESTION_IDS = {
    "q001", "q002", "q003", "q004", "q005", "q006", "q007", "q008", "q009", "q010",
    "q011", "q012", "q013", "q014", "q015", "q016", "q017", "q018", "q019", "q020",
    "q021", "q022", "q031", "q036", "q037", "q038", "q039", "q040", "q041", "q042", "q045",
}


def matches_expected(cited_id: str, expected_set: set[str]) -> bool:
    """Prefix-match cited article IDs against expected IDs."""
    return any(cited_id.startswith(exp) for exp in expected_set)


def _matches_expected_source(cited_source: dict, expected_source: dict) -> bool:
    """Check precise law-aware match with prefix-compatible article ID semantics."""
    cited_law = (cited_source.get("law_id") or "").strip()
    cited_article = (cited_source.get("article_id") or "").strip()
    expected_law = (expected_source.get("law_id") or "").strip()
    expected_article = (expected_source.get("article_id") or "").strip()
    if not expected_law or not expected_article:
        return False
    return cited_law == expected_law and cited_article.startswith(expected_article)


def infer_negative_type(row: dict) -> str:
    """Infer negative subtype from explicit fields or note suffix."""
    explicit = (row.get("negative_type") or "").strip().lower()
    if explicit:
        return explicit

    category = (row.get("category") or "").strip().lower()
    if category in {"ambiguity", "offtopic_legal", "offtopic_nonlegal"}:
        return category

    notes = (row.get("notes") or "")
    match = re.search(r"negative_type:\s*([a-z_]+)", notes, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return "unknown"


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


def collect_indexed_article_ids(chain: LegalRAGChain) -> set[str] | None:
    """
    Best-effort retrieval of article IDs from Qdrant payloads.

    Returns None if unavailable (for example due permissions/network).
    """
    client = getattr(chain.vectorstore, "client", None)
    if client is None:
        return None

    article_ids: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=chain.settings.qdrant_collection_name,
                limit=512,
                offset=offset,
                with_payload=["article_id"],
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                article_id = payload.get("article_id")
                if article_id:
                    article_ids.add(article_id)
            if offset is None:
                break
    except Exception as exc:  # pragma: no cover - depends on remote DB state
        logger.warning("Skipped indexed article ID scan: %s", exc)
        return None

    return article_ids


def validate_questions(
    questions: list[dict],
    chain: LegalRAGChain,
    *,
    skip_index_scan: bool = False,
) -> None:
    """Run lightweight sanity checks on the eval set."""
    # 1) Hard fail on empty question text.
    empty_ids = [row.get("id", "unknown") for row in questions if not (row.get("question") or "").strip()]
    if empty_ids:
        raise ValueError(f"Found empty question text for IDs: {', '.join(empty_ids)}")

    # 2) Warn on skewed categories.
    categories = [row.get("category", "uncategorized") for row in questions]
    counts = Counter(categories)
    total = max(len(questions), 1)
    largest_category, largest_count = counts.most_common(1)[0]
    if (largest_count / total) > 0.40:
        logger.warning(
            "Category skew detected: '%s' is %.1f%% of dataset (%s/%s)",
            largest_category,
            100.0 * largest_count / total,
            largest_count,
            total,
        )

    # 3) Warn when expected article IDs don't seem present in indexed corpus.
    if skip_index_scan:
        logger.info("Skipped indexed corpus validation scan (skip_index_scan=true).")
        return
    indexed_ids = collect_indexed_article_ids(chain)
    if indexed_ids is None:
        logger.warning("Could not validate expected_articles against indexed corpus; continuing.")
        return

    unique_expected = {
        expected_id
        for row in questions
        for expected_id in row.get("expected_articles", [])
    }
    missing = [
        exp
        for exp in sorted(unique_expected)
        if not any(indexed.startswith(exp) for indexed in indexed_ids)
    ]
    if missing:
        logger.warning(
            "Expected article IDs not found in index (%s): %s",
            len(missing),
            ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""),
        )


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
        docs = chain._invoke_retriever(query, routed_law_ids=chain._route_law_ids(query))

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
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            scores = [float(s) for s in raw_scores]
            # Convert logits to [0, 1] consistently with runtime pipeline.
            normalized = [1.0 / (1.0 + math.exp(-s)) for s in scores]
            for doc, score in zip(docs, normalized):
                metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
                candidates.append(
                    {
                        "law_id": metadata.get("law_id", ""),
                        "article_id": metadata.get("article_id", ""),
                        "doc_type": metadata.get("doc_type", "provision"),
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
        if len(kept) < min(min_sources, len(ranked)):
            kept = ranked[: min(min_sources, len(ranked))]

        # Simulate runtime doc-type prioritization and editorial budgeting.
        provisions = [c for c in kept if c.get("doc_type", "provision") != "editorial_note"]
        editorial = [c for c in kept if c.get("doc_type", "provision") == "editorial_note"]
        editorial_budget = chain._compute_editorial_budget(total_docs=len(kept), retrieval_query=question)
        kept = provisions + editorial[:editorial_budget]

        scores = [c["score"] for c in kept]
        cited_ids = [c["article_id"] for c in kept if c.get("article_id")]
        cited_sources = [
            {"law_id": c.get("law_id", ""), "article_id": c.get("article_id", "")}
            for c in kept
            if c.get("article_id")
        ]
        cited_editorial = any(
            (c.get("doc_type", "provision") == "editorial_note")
            for c in kept
        )
        top_score = scores[0] if scores else None

        if top_score is not None:
            top_scores.append(top_score)

        if expected:
            retrieval_total += 1
            if expected_sources:
                matched_expected_pairs = set()
                matched_citations = 0
                for cited in cited_sources:
                    hit = False
                    for exp in expected_sources:
                        if _matches_expected_source(cited, exp):
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
        else:
            # Ambiguous / off-topic guardrail: ideally no sources, or gated.
            ambiguity_total += 1
            is_gated = False
            if top_score is not None and top_score < confidence_threshold:
                is_gated = True
            if (
                not is_gated
                and chain.settings.reranker_ambiguity_gating_enabled
                and len(scores) >= 2
                and scores[0] <= top_score_ceiling
            ):
                if (scores[0] - scores[1]) < ambiguity_min_gap:
                    is_gated = True
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

    return {
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
        "editorial_context_success": editorial_context_success,
        "core_editorial_context_success": core_editorial_context_success,
        "non_editorial_clean_success": non_editorial_clean_success,
        "core_non_editorial_clean_success": core_non_editorial_clean_success,
    }


def apply_combo_to_chain(
    chain: LegalRAGChain,
    retrieval_k_initial: int,
    confidence: float,
    min_doc: float,
    min_gap: float,
    top_score_ceiling: float,
) -> None:
    """Apply sweep parameters to an existing chain instance."""
    chain.settings.retrieval_k_initial = retrieval_k_initial
    chain.settings.reranker_confidence_threshold = confidence
    chain.settings.reranker_min_doc_score = min_doc
    chain.settings.reranker_ambiguity_min_gap = min_gap
    chain.settings.reranker_ambiguity_top_score_ceiling = top_score_ceiling


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

    retrieval_k_initial_values = [15, 20]
    confidence_values = [0.35, 0.45]
    min_doc_values = [0.25, 0.35]
    min_gap_values = [0.05, 0.10]
    top_score_ceiling_values = [0.60, 0.70]

    logger.info("Loaded %s questions", len(questions))
    core_count = sum(1 for row in questions if row.get("id") in CORE_QUESTION_IDS)
    logger.info("Frozen core subset size: %s", core_count)
    logger.info("Starting retrieval sweep...")
    chain = LegalRAGChain(settings=settings)
    skip_index_scan = _env_flag("SWEEP_SKIP_INDEX_SCAN", default=True)
    validate_questions(questions, chain, skip_index_scan=skip_index_scan)
    max_k_initial = max(retrieval_k_initial_values)
    logger.info("Precomputing candidates once with k=%s...", max_k_initial)
    cached_candidates = precompute_candidates(chain, questions, max_k_initial=max_k_initial)

    rows: list[dict] = []
    for retrieval_k_initial, confidence, min_doc, min_gap, top_score_ceiling in itertools.product(
        retrieval_k_initial_values,
        confidence_values,
        min_doc_values,
        min_gap_values,
        top_score_ceiling_values,
    ):
        apply_combo_to_chain(
            chain, retrieval_k_initial, confidence, min_doc, min_gap, top_score_ceiling
        )
        metrics = evaluate_combo(
            chain,
            cached_candidates,
            retrieval_k_initial=retrieval_k_initial,
            confidence_threshold=confidence,
            min_doc_score=min_doc,
            ambiguity_min_gap=min_gap,
            top_score_ceiling=top_score_ceiling,
        )
        row = {
            "retrieval_k_initial": retrieval_k_initial,
            "reranker_confidence_threshold": confidence,
            "reranker_min_doc_score": min_doc,
            "reranker_ambiguity_min_gap": min_gap,
            "reranker_ambiguity_top_score_ceiling": top_score_ceiling,
            **metrics,
        }
        rows.append(row)
        logger.info(
            "k_init=%s conf=%.2f min_doc=%.2f min_gap=%.2f top_ceiling=%.2f -> "
            "recall=%.3f coverage=%.3f precision=%.3f unexpected=%.3f "
            "single=%.3f multi=%.3f neg=%.3f neg_legal=%.3f neg_nonlegal=%.3f "
            "editorial=%.3f non_editorial_clean=%.3f avg_top=%.3f",
            retrieval_k_initial,
            confidence,
            min_doc,
            min_gap,
            top_score_ceiling,
            metrics["recall_at_k"],
            metrics["mean_expected_coverage"],
            metrics["citation_precision"],
            metrics["unexpected_citation_rate"],
            metrics["single_article_recall_at_k"],
            metrics["multi_article_recall_at_k"],
            metrics["negative_success"],
            metrics["negative_offtopic_legal_success"],
            metrics["negative_offtopic_nonlegal_success"],
            metrics["editorial_context_success"],
            metrics["non_editorial_clean_success"],
            metrics["avg_top_score"],
        )

    # Sort by leakage-first objective, then coverage/recall.
    rows.sort(
        key=lambda r: (
            r["negative_offtopic_nonlegal_success"],
            r["negative_offtopic_legal_success"],
            r["negative_ambiguity_success"],
            r["non_editorial_clean_success"],
            r["recall_at_k"],
            r["mean_expected_coverage"],
            r["citation_precision"],
            -r["unexpected_citation_rate"],
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
