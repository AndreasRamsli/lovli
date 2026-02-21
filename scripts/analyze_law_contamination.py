#!/usr/bin/env python3
"""
Analyze cross-law contamination in retrieval results.

The script runs retrieval/reranking for each eval question and reports:
- Per-question law distribution and score summaries
- Aggregate contamination metrics across the dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import hashlib
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain  # noqa: E402
from lovli.config import get_settings  # noqa: E402
from lovli.scoring import matches_expected_source  # noqa: E402
from lovli.profiles import apply_trust_profile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
INTENT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
INTENT_TERMS = {
    "husleie",
    "depositum",
    "oppsigelse",
    "oppsigelsesfrist",
    "leie",
    "utleier",
    "leietaker",
}


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


def _contamination_cache_key(
    questions_sha256: str,
    git_commit: str,
    trust_profile: str,
    profile_version: str,
    retrieval_k_initial: int,
    qdrant_collection: str,
    question_count: int,
) -> str:
    """Deterministic cache key for contamination report. Invalidation on any input change."""
    blob = (
        f"{questions_sha256}|{git_commit}|{trust_profile}|{profile_version}|"
        f"{retrieval_k_initial}|{qdrant_collection}|{question_count}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load jsonl questions."""
    questions: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _extract_intent_terms(question: str) -> list[str]:
    """Extract simple lexical intent terms for failure clustering."""
    tokens = set(INTENT_TOKEN_RE.findall((question or "").lower()))
    return sorted(tokens & INTENT_TERMS)


def analyze_question(
    chain: LegalRAGChain, row: dict[str, Any], retrieval_k_initial: int
) -> dict[str, Any]:
    """Analyze one question and return law-level contamination diagnostics."""
    question = row["question"]
    expected_sources = row.get("expected_sources", []) or []
    expected_articles = row.get("expected_articles", []) or []
    expected_law_ids = sorted(
        {
            (item.get("law_id") or "").strip()
            for item in expected_sources
            if (item.get("law_id") or "").strip()
        }
    )

    routed_law_ids = chain._route_law_ids(question)
    routing_diagnostics = dict(getattr(chain, "_last_routing_diagnostics", {}) or {})
    docs = chain._invoke_retriever(question, routed_law_ids=routed_law_ids)
    routing_diagnostics = dict(
        getattr(chain, "_last_routing_diagnostics", routing_diagnostics) or routing_diagnostics
    )

    # Deduplicate by (law_id, article_id) the same way as runtime retrieve().
    dedup_docs = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        key = (metadata.get("law_id"), metadata.get("article_id"))
        if metadata.get("article_id") and key in seen:
            continue
        if metadata.get("article_id"):
            seen.add(key)
        dedup_docs.append(doc)
    docs = dedup_docs[:retrieval_k_initial]

    # Apply reranking + doc score floor to match retrieval behavior.
    reranked_docs, scores = chain._rerank(question, docs, top_k=chain.settings.retrieval_k)
    reranked_docs, scores = chain._apply_reranker_doc_filter(reranked_docs, scores)
    if chain.settings.law_coherence_filter_enabled:
        reranked_docs, scores = chain._filter_by_law_coherence(reranked_docs, scores)
    coherence_diagnostics = dict(getattr(chain, "_last_coherence_diagnostics", {}) or {})

    # Keep provision docs only in diagnostics.
    provision_rows: list[dict[str, Any]] = []
    for doc, score in zip(reranked_docs, scores):
        metadata = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
        if (metadata.get("doc_type") or "provision") == "editorial_note":
            continue
        provision_rows.append(
            {
                "law_id": metadata.get("law_id", ""),
                "law_title": metadata.get("law_title", ""),
                "article_id": metadata.get("article_id", ""),
                "score": float(score),
            }
        )

    # Law-level grouping
    by_law: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in provision_rows:
        by_law[item["law_id"]].append(item)

    law_stats: list[dict[str, Any]] = []
    for law_id, items in by_law.items():
        law_scores = [it["score"] for it in items]
        law_stats.append(
            {
                "law_id": law_id,
                "law_title": items[0]["law_title"] if items else "",
                "count": len(items),
                "avg_score": mean(law_scores) if law_scores else 0.0,
                "max_score": max(law_scores) if law_scores else 0.0,
                "articles": [it["article_id"] for it in items],
            }
        )
    law_stats.sort(key=lambda item: (item["count"], item["avg_score"]), reverse=True)

    dominant_law = law_stats[0]["law_id"] if law_stats else None
    dominant_avg_score = law_stats[0]["avg_score"] if law_stats else None
    foreign_laws = [entry for entry in law_stats[1:]]

    has_contamination = len(law_stats) > 1
    singleton_foreign_laws = sum(1 for entry in foreign_laws if entry["count"] == 1)
    foreign_score_gaps = []
    if dominant_avg_score is not None:
        for entry in foreign_laws:
            foreign_score_gaps.append(dominant_avg_score - entry["avg_score"])

    cited_sources = [
        {"law_id": item["law_id"], "article_id": item["article_id"]} for item in provision_rows
    ]
    unexpected_sources = []
    matched_expected_pairs = 0
    if expected_sources:
        for cited in cited_sources:
            if not any(matches_expected_source(cited, exp) for exp in expected_sources):
                unexpected_sources.append(cited)
        matched_expected_pairs = sum(
            1
            for exp in expected_sources
            if any(matches_expected_source(cited, exp) for cited in cited_sources)
        )
    else:
        # For negatives, all citations are unexpected.
        unexpected_sources = cited_sources

    routed_law_ids_normalized = [law_id for law_id in routed_law_ids if (law_id or "").strip()]
    route_miss_expected_law = bool(
        expected_law_ids
        and routed_law_ids_normalized
        and set(expected_law_ids).isdisjoint(set(routed_law_ids_normalized))
    )
    dominant_law_mismatch = bool(
        expected_law_ids and dominant_law and dominant_law not in set(expected_law_ids)
    )
    expected_law_matched_in_dominant = bool(
        expected_law_ids and dominant_law and dominant_law in set(expected_law_ids)
    )
    expected_source_retrieved = bool(matched_expected_pairs > 0)
    effective_expected_law_miss = bool(
        expected_law_ids and not expected_law_matched_in_dominant and not expected_source_retrieved
    )
    fallback_reason = (routing_diagnostics.get("retrieval_fallback") or "").strip()
    fallback_stage = (routing_diagnostics.get("retrieval_fallback_stage") or "none").strip()
    fallback_triggered = bool(fallback_reason)
    fallback_recovered = bool(fallback_triggered and matched_expected_pairs > 0)
    routing_confidence = routing_diagnostics.get("routing_confidence") or {}
    routing_selection_mode = (routing_confidence.get("selection_mode") or "unknown").strip()
    routing_mode = (routing_confidence.get("mode") or "unknown").strip()

    return {
        "id": row.get("id"),
        "question": question,
        "expected_articles": expected_articles,
        "expected_sources": expected_sources,
        "retrieved_sources_count": len(cited_sources),
        "distinct_law_count": len(law_stats),
        "dominant_law_id": dominant_law,
        "has_contamination": has_contamination,
        "singleton_foreign_laws": singleton_foreign_laws,
        "avg_foreign_score_gap": mean(foreign_score_gaps) if foreign_score_gaps else None,
        "matched_expected_source_count": matched_expected_pairs,
        "unexpected_sources_count": len(unexpected_sources),
        "expected_law_ids": expected_law_ids,
        "route_miss_expected_law": route_miss_expected_law,
        "dominant_law_mismatch": dominant_law_mismatch,
        "expected_law_matched_in_dominant": expected_law_matched_in_dominant,
        "expected_source_retrieved": expected_source_retrieved,
        "effective_expected_law_miss": effective_expected_law_miss,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason or None,
        "fallback_stage": fallback_stage or None,
        "fallback_recovered": fallback_recovered,
        "routing_selection_mode": routing_selection_mode,
        "routing_mode": routing_mode,
        "intent_terms": _extract_intent_terms(question),
        "routed_law_ids": routed_law_ids,
        "routing_diagnostics": routing_diagnostics,
        "coherence_diagnostics": coherence_diagnostics,
        "law_stats": law_stats,
        "unexpected_sources": unexpected_sources,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create aggregate contamination summary."""
    total = len(results)
    contaminated = [r for r in results if r["has_contamination"]]
    singleton_foreign_cases = [r for r in results if (r.get("singleton_foreign_laws") or 0) > 0]

    gaps = [
        r["avg_foreign_score_gap"] for r in results if r.get("avg_foreign_score_gap") is not None
    ]
    unexpected_total = sum(r.get("unexpected_sources_count", 0) for r in results)
    cited_total = sum(r.get("retrieved_sources_count", 0) for r in results)
    positives = [r for r in results if (r.get("expected_sources") or [])]

    def _effective_expected_law_miss(row: dict[str, Any]) -> bool:
        """Compute post-fallback miss; supports legacy rows without explicit field."""
        explicit = row.get("effective_expected_law_miss")
        if explicit is not None:
            return bool(explicit)
        expected_law_ids = set(row.get("expected_law_ids") or [])
        if not expected_law_ids:
            expected_law_ids = {
                (item.get("law_id") or "").strip()
                for item in (row.get("expected_sources") or [])
                if (item.get("law_id") or "").strip()
            }
        if not expected_law_ids:
            return False
        dominant_law = (row.get("dominant_law_id") or "").strip()
        expected_source_retrieved = row.get("expected_source_retrieved")
        if expected_source_retrieved is None:
            matched_expected_source_count = int(row.get("matched_expected_source_count") or 0)
            expected_source_retrieved = bool(matched_expected_source_count > 0)
        else:
            expected_source_retrieved = bool(expected_source_retrieved)
        expected_law_matched_in_dominant = bool(dominant_law and dominant_law in expected_law_ids)
        return bool(not expected_law_matched_in_dominant and not expected_source_retrieved)

    # Backward-compatible candidate-routing miss metric (pre-fallback semantics).
    route_miss_cases = [r for r in positives if r.get("route_miss_expected_law")]
    # Effective miss metric (post-fallback semantics): expected law/source still absent in final retrieval.
    effective_expected_law_miss_cases = [r for r in positives if _effective_expected_law_miss(r)]
    recovered_from_route_miss_cases = [
        r for r in route_miss_cases if not _effective_expected_law_miss(r)
    ]
    dominant_mismatch_cases = [r for r in positives if r.get("dominant_law_mismatch")]
    fallback_cases = [r for r in results if r.get("fallback_triggered")]
    fallback_recovered_cases = [r for r in fallback_cases if r.get("fallback_recovered")]
    fallback_stage_counts: dict[str, int] = defaultdict(int)
    routing_selection_mode_counts: dict[str, int] = defaultdict(int)
    route_miss_by_fallback_stage: dict[str, int] = defaultdict(int)
    route_miss_by_routing_mode: dict[str, int] = defaultdict(int)
    effective_miss_by_fallback_stage: dict[str, int] = defaultdict(int)
    effective_miss_by_routing_mode: dict[str, int] = defaultdict(int)
    fallback_triggered_by_stage: dict[str, int] = defaultdict(int)
    fallback_recovered_by_stage: dict[str, int] = defaultdict(int)
    positive_count_by_stage: dict[str, int] = defaultdict(int)
    route_miss_count_by_mode_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    effective_miss_count_by_mode_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    positive_count_by_mode_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    route_miss_law_pair_confusions: dict[str, int] = defaultdict(int)
    effective_miss_law_pair_confusions: dict[str, int] = defaultdict(int)
    for row in route_miss_cases:
        expected = ",".join(sorted(row.get("expected_law_ids") or [])) or "__none__"
        dominant = (row.get("dominant_law_id") or "__none__").strip()
        route_miss_law_pair_confusions[f"{expected}->{dominant}"] += 1
        fallback_stage = (row.get("fallback_stage") or "none").strip()
        routing_mode = (row.get("routing_selection_mode") or "unknown").strip()
        route_miss_by_fallback_stage[fallback_stage] += 1
        route_miss_by_routing_mode[routing_mode] += 1
        route_miss_count_by_mode_stage[routing_mode][fallback_stage] += 1
    for row in effective_expected_law_miss_cases:
        expected = ",".join(sorted(row.get("expected_law_ids") or [])) or "__none__"
        dominant = (row.get("dominant_law_id") or "__none__").strip()
        effective_miss_law_pair_confusions[f"{expected}->{dominant}"] += 1
        fallback_stage = (row.get("fallback_stage") or "none").strip()
        routing_mode = (row.get("routing_selection_mode") or "unknown").strip()
        effective_miss_by_fallback_stage[fallback_stage] += 1
        effective_miss_by_routing_mode[routing_mode] += 1
        effective_miss_count_by_mode_stage[routing_mode][fallback_stage] += 1
    for row in results:
        fallback_stage = (row.get("fallback_stage") or "none").strip()
        routing_mode = (row.get("routing_selection_mode") or "unknown").strip()
        fallback_stage_counts[fallback_stage] += 1
        routing_selection_mode_counts[routing_mode] += 1
        if row.get("fallback_triggered"):
            fallback_triggered_by_stage[fallback_stage] += 1
            if row.get("fallback_recovered"):
                fallback_recovered_by_stage[fallback_stage] += 1
        if row.get("expected_sources"):
            positive_count_by_stage[fallback_stage] += 1
            positive_count_by_mode_stage[routing_mode][fallback_stage] += 1

    fallback_recovery_rate_by_stage = {
        stage: (float(fallback_recovered_by_stage.get(stage, 0)) / float(count) if count else 0.0)
        for stage, count in sorted(fallback_triggered_by_stage.items())
    }
    route_miss_rate_by_stage = {
        stage: (float(route_miss_by_fallback_stage.get(stage, 0)) / float(count) if count else 0.0)
        for stage, count in sorted(positive_count_by_stage.items())
    }
    route_miss_rate_by_mode_stage = {
        mode: {
            stage: (
                float(route_miss_count_by_mode_stage.get(mode, {}).get(stage, 0))
                / float(positive_count)
                if positive_count
                else 0.0
            )
            for stage, positive_count in sorted(stage_counts.items())
        }
        for mode, stage_counts in sorted(positive_count_by_mode_stage.items())
    }
    effective_miss_rate_by_stage = {
        stage: (
            float(effective_miss_by_fallback_stage.get(stage, 0)) / float(count) if count else 0.0
        )
        for stage, count in sorted(positive_count_by_stage.items())
    }
    effective_miss_rate_by_mode_stage = {
        mode: {
            stage: (
                float(effective_miss_count_by_mode_stage.get(mode, {}).get(stage, 0))
                / float(positive_count)
                if positive_count
                else 0.0
            )
            for stage, positive_count in sorted(stage_counts.items())
        }
        for mode, stage_counts in sorted(positive_count_by_mode_stage.items())
    }

    failing_intent_counts: dict[str, int] = defaultdict(int)
    for row in route_miss_cases:
        for term in row.get("intent_terms") or []:
            failing_intent_counts[term] += 1
    failing_intent_clusters = sorted(
        ({"intent_term": term, "count": count} for term, count in failing_intent_counts.items()),
        key=lambda item: item["count"],
        reverse=True,
    )

    return {
        "total_questions": total,
        "contamination_case_count": len(contaminated),
        "contamination_rate": (len(contaminated) / total) if total else 0.0,
        "singleton_foreign_case_count": len(singleton_foreign_cases),
        "singleton_foreign_rate": (len(singleton_foreign_cases) / total) if total else 0.0,
        "mean_foreign_score_gap": mean(gaps) if gaps else None,
        "unexpected_citation_rate": (unexpected_total / cited_total) if cited_total else 0.0,
        "total_unexpected_sources": unexpected_total,
        "total_cited_sources": cited_total,
        "routing_fallback_count": sum(
            1
            for r in results
            if ((r.get("routing_diagnostics") or {}).get("retrieval_fallback") is not None)
        ),
        "coherence_filtered_query_count": sum(
            1
            for r in results
            if (((r.get("coherence_diagnostics") or {}).get("removed_count") or 0) > 0)
        ),
        "route_miss_expected_law_count": len(route_miss_cases),
        "route_miss_expected_law_rate": (len(route_miss_cases) / len(positives))
        if positives
        else 0.0,
        "routing_candidate_miss_count": len(route_miss_cases),
        "routing_candidate_miss_rate": (len(route_miss_cases) / len(positives))
        if positives
        else 0.0,
        "effective_expected_law_miss_count": len(effective_expected_law_miss_cases),
        "effective_expected_law_miss_rate": (
            len(effective_expected_law_miss_cases) / len(positives)
        )
        if positives
        else 0.0,
        "route_miss_recovered_count": len(recovered_from_route_miss_cases),
        "route_miss_recovered_rate": (len(recovered_from_route_miss_cases) / len(route_miss_cases))
        if route_miss_cases
        else 0.0,
        "dominant_law_mismatch_count": len(dominant_mismatch_cases),
        "dominant_law_mismatch_rate": (len(dominant_mismatch_cases) / len(positives))
        if positives
        else 0.0,
        "fallback_triggered_count": len(fallback_cases),
        "fallback_recovered_count": len(fallback_recovered_cases),
        "fallback_recovery_rate": (len(fallback_recovered_cases) / len(fallback_cases))
        if fallback_cases
        else 0.0,
        "fallback_stage_counts": dict(sorted(fallback_stage_counts.items())),
        "routing_selection_mode_counts": dict(sorted(routing_selection_mode_counts.items())),
        "route_miss_by_fallback_stage": dict(sorted(route_miss_by_fallback_stage.items())),
        "route_miss_by_routing_mode": dict(sorted(route_miss_by_routing_mode.items())),
        "effective_miss_by_fallback_stage": dict(sorted(effective_miss_by_fallback_stage.items())),
        "effective_miss_by_routing_mode": dict(sorted(effective_miss_by_routing_mode.items())),
        "fallback_recovery_rate_by_stage": fallback_recovery_rate_by_stage,
        "route_miss_rate_by_stage": route_miss_rate_by_stage,
        "effective_miss_rate_by_stage": effective_miss_rate_by_stage,
        "route_miss_count_by_mode_stage": {
            mode: dict(sorted(stage_counts.items()))
            for mode, stage_counts in sorted(route_miss_count_by_mode_stage.items())
        },
        "route_miss_rate_by_mode_stage": route_miss_rate_by_mode_stage,
        "effective_miss_count_by_mode_stage": {
            mode: dict(sorted(stage_counts.items()))
            for mode, stage_counts in sorted(effective_miss_count_by_mode_stage.items())
        },
        "effective_miss_rate_by_mode_stage": effective_miss_rate_by_mode_stage,
        "top_route_miss_law_pair_confusions": [
            {"law_pair": pair, "count": count}
            for pair, count in sorted(
                route_miss_law_pair_confusions.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ],
        "top_effective_miss_law_pair_confusions": [
            {"law_pair": pair, "count": count}
            for pair, count in sorted(
                effective_miss_law_pair_confusions.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ],
        "top_failing_intent_clusters": failing_intent_clusters[:10],
    }


def build_hard_cluster_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build focused diagnostics for mismatch/fallback-heavy hard clusters."""
    if not results:
        return {"cluster_count": 0, "clusters": []}

    cluster_rows: dict[str, list[dict[str, Any]]] = {}
    mismatch_rows = [row for row in results if row.get("dominant_law_mismatch")]
    if mismatch_rows:
        cluster_rows["dominant_law_mismatch"] = mismatch_rows

    fallback_rows = [
        row for row in results if (row.get("fallback_reason") or "").startswith("uncertainty")
    ]
    if fallback_rows:
        cluster_rows["uncertainty_fallback"] = fallback_rows

    # Auto-cluster by repeated (expected laws -> dominant law) mismatch pairs.
    pair_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mismatch_rows:
        expected = ",".join(sorted(row.get("expected_law_ids") or [])) or "__none__"
        dominant = (row.get("dominant_law_id") or "__none__").strip()
        pair_buckets[f"law_pair:{expected}->{dominant}"].append(row)
    for key, rows in pair_buckets.items():
        if len(rows) >= 2:
            cluster_rows[key] = rows

    clusters: list[dict[str, Any]] = []
    for cluster_id, rows in cluster_rows.items():
        fallback_reason_dist: dict[str, int] = defaultdict(int)
        mismatch_count = 0
        unexpected_count = 0
        cited_count = 0
        for row in rows:
            fallback_reason = (row.get("fallback_reason") or "none").strip()
            fallback_reason_dist[fallback_reason] += 1
            mismatch_count += 1 if row.get("dominant_law_mismatch") else 0
            unexpected_count += int(row.get("unexpected_sources_count") or 0)
            cited_count += int(row.get("retrieved_sources_count") or 0)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "question_count": len(rows),
                "question_ids": [row.get("id") for row in rows],
                "fallback_reason_distribution": dict(sorted(fallback_reason_dist.items())),
                "dominant_law_mismatch_rate": (mismatch_count / len(rows)) if rows else 0.0,
                "unexpected_citation_contribution": (
                    unexpected_count / cited_count if cited_count else 0.0
                ),
            }
        )
    clusters.sort(key=lambda item: item["question_count"], reverse=True)
    return {"cluster_count": len(clusters), "clusters": clusters}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cross-law contamination in retrieval results."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT_DIR / "eval" / "questions.jsonl",
        help="Path to eval questions JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "eval" / "law_contamination_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional first-N sample size for faster iteration.",
    )
    parser.add_argument(
        "--retrieval-k-initial",
        type=int,
        default=None,
        help="Override retrieval_k_initial for this analysis run.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    settings = get_settings()
    profile_name = os.getenv("TRUST_PROFILE", settings.trust_profile_name)
    resolved_profile = apply_trust_profile(settings, profile_name)
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = (
        os.getenv("LOVLI_RUN_ID")
        or f"contam_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    git_commit = _safe_git_commit()
    questions_sha256 = _sha256_file(args.questions)
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
    if args.retrieval_k_initial is not None and args.retrieval_k_initial > 0:
        settings.retrieval_k_initial = args.retrieval_k_initial
    retrieval_k_initial = settings.retrieval_k_initial

    questions = load_questions(args.questions)
    sample_size = args.sample_size
    if sample_size <= 0:
        raw = os.getenv("CONTAMINATION_SAMPLE_SIZE")
        if raw:
            try:
                sample_size = int(raw)
            except ValueError:
                sample_size = 0
    if sample_size > 0:
        questions = questions[:sample_size]
        logger.info("Using sample size: %s (CONTAMINATION_SAMPLE_SIZE)", sample_size)

    cache_dir_raw = os.getenv("CONTAMINATION_CACHE_DIR")
    cache_dir = Path(cache_dir_raw) if cache_dir_raw and cache_dir_raw.strip() else None
    key_hash = _contamination_cache_key(
        questions_sha256,
        git_commit,
        settings.trust_profile_name,
        settings.trust_profile_version,
        retrieval_k_initial,
        settings.qdrant_collection_name,
        len(questions),
    )
    cache_path = cache_dir / f"contamination_report_{key_hash}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            meta = cached.get("artifact_metadata") or {}
            if (
                meta.get("questions_sha256") == questions_sha256
                and meta.get("git_commit") == git_commit
                and meta.get("question_count") == len(questions)
            ):
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(cached, f, ensure_ascii=False, indent=2)
                agg = cached.get("aggregate", {})
                logger.info("Loaded contamination report from cache %s", key_hash[:12])
                logger.info(
                    "Contamination rate=%.1f%%, singleton_foreign_rate=%.1f%%, unexpected_citation_rate=%.1f%%",
                    agg.get("contamination_rate", 0) * 100,
                    agg.get("singleton_foreign_rate", 0) * 100,
                    agg.get("unexpected_citation_rate", 0) * 100,
                )
                return
        except Exception as e:
            logger.warning("Contamination cache load failed: %s", e)

    chain = LegalRAGChain(settings=settings)
    logger.info(
        "Analyzing %s questions (retrieval_k_initial=%s)", len(questions), retrieval_k_initial
    )

    workers_raw = os.getenv("CONTAMINATION_PARALLEL_WORKERS", "0").strip()
    try:
        parallel_workers = max(0, int(workers_raw))
    except ValueError:
        parallel_workers = 0

    retrieval_k = retrieval_k_initial
    if parallel_workers > 0:
        rows = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_idx = {
                executor.submit(analyze_question, chain, row, retrieval_k_initial=retrieval_k): idx
                for idx, row in enumerate(questions)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                rows[idx] = future.result()
                done += 1
                if done % 10 == 0 or done == len(questions):
                    logger.info("Processed %s/%s questions", done, len(questions))
    else:
        rows = []
        for idx, row in enumerate(questions, start=1):
            rows.append(analyze_question(chain, row, retrieval_k_initial=retrieval_k))
            if idx % 10 == 0 or idx == len(questions):
                logger.info("Processed %s/%s questions", idx, len(questions))

    aggregate = summarize(rows)
    hard_cluster_summary = build_hard_cluster_summary(rows)
    artifact_metadata = {
        "artifact_type": "law_contamination_report",
        "run_id": run_id,
        "run_started_at": run_started_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "trust_profile_name": settings.trust_profile_name,
        "trust_profile_version": settings.trust_profile_version,
        "questions_path": str(args.questions),
        "questions_sha256": questions_sha256,
        "question_count": len(questions),
    }
    gate_summary = {
        # Legacy gate metric (pre-fallback candidate semantics). Keep during migration.
        "route_miss_expected_law_rate": aggregate.get("route_miss_expected_law_rate"),
        "routing_candidate_miss_rate": aggregate.get("routing_candidate_miss_rate"),
        "effective_expected_law_miss_rate": aggregate.get("effective_expected_law_miss_rate"),
        "dominant_law_mismatch_rate": aggregate.get("dominant_law_mismatch_rate"),
        "unexpected_citation_rate": aggregate.get("unexpected_citation_rate"),
        "fallback_recovery_rate": aggregate.get("fallback_recovery_rate"),
        "fallback_triggered_count": aggregate.get("fallback_triggered_count"),
    }
    report = {
        "trust_profile_name": settings.trust_profile_name,
        "trust_profile_version": settings.trust_profile_version,
        "artifact_metadata": artifact_metadata,
        "gate_summary": gate_summary,
        "aggregate": aggregate,
        "hard_cluster_summary": hard_cluster_summary,
        "per_question": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if cache_path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("Saved contamination cache %s", key_hash[:12])
        except Exception as e:
            logger.warning("Contamination cache save failed: %s", e)

    logger.info("Saved contamination report to %s", args.output)
    logger.info(
        "Contamination rate=%.1f%%, singleton_foreign_rate=%.1f%%, unexpected_citation_rate=%.1f%%",
        aggregate["contamination_rate"] * 100.0,
        aggregate["singleton_foreign_rate"] * 100.0,
        aggregate["unexpected_citation_rate"] * 100.0,
    )
    logger.info(
        "Gate summary: candidate_route_miss=%.4f effective_route_miss=%.4f "
        "dominant_mismatch=%.4f unexpected=%.4f fallback_recovery=%.4f fallback_count=%s",
        float(gate_summary["route_miss_expected_law_rate"] or 0.0),
        float(gate_summary["effective_expected_law_miss_rate"] or 0.0),
        float(gate_summary["dominant_law_mismatch_rate"] or 0.0),
        float(gate_summary["unexpected_citation_rate"] or 0.0),
        float(gate_summary["fallback_recovery_rate"] or 0.0),
        int(gate_summary["fallback_triggered_count"] or 0),
    )
    logger.info(
        "Compatibility note: route_miss_expected_law_rate uses pre-fallback routing-candidate semantics. "
        "Use effective_expected_law_miss_rate for post-fallback gate decisions."
    )


if __name__ == "__main__":
    # Keep local analysis runs out of LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    main()
