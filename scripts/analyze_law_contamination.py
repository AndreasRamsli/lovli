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
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain  # noqa: E402
from lovli.config import get_settings  # noqa: E402
from lovli.retrieval_shared import matches_expected_source  # noqa: E402
from lovli.trust_profiles import apply_trust_profile  # noqa: E402

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


def analyze_question(chain: LegalRAGChain, row: dict[str, Any], retrieval_k_initial: int) -> dict[str, Any]:
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
    routing_diagnostics = dict(getattr(chain, "_last_routing_diagnostics", routing_diagnostics) or routing_diagnostics)

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

    cited_sources = [{"law_id": item["law_id"], "article_id": item["article_id"]} for item in provision_rows]
    unexpected_sources = []
    matched_expected_pairs = 0
    if expected_sources:
        for cited in cited_sources:
            if not any(matches_expected_source(cited, exp) for exp in expected_sources):
                unexpected_sources.append(cited)
        matched_expected_pairs = sum(
            1 for exp in expected_sources if any(matches_expected_source(cited, exp) for cited in cited_sources)
        )
    else:
        # For negatives, all citations are unexpected.
        unexpected_sources = cited_sources

    routed_law_ids_normalized = [
        law_id for law_id in routed_law_ids if (law_id or "").strip()
    ]
    route_miss_expected_law = bool(
        expected_law_ids
        and routed_law_ids_normalized
        and set(expected_law_ids).isdisjoint(set(routed_law_ids_normalized))
    )
    dominant_law_mismatch = bool(
        expected_law_ids and dominant_law and dominant_law not in set(expected_law_ids)
    )
    fallback_reason = (routing_diagnostics.get("retrieval_fallback") or "").strip()
    fallback_triggered = bool(fallback_reason)
    fallback_recovered = bool(fallback_triggered and matched_expected_pairs > 0)

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
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason or None,
        "fallback_recovered": fallback_recovered,
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

    gaps = [r["avg_foreign_score_gap"] for r in results if r.get("avg_foreign_score_gap") is not None]
    unexpected_total = sum(r.get("unexpected_sources_count", 0) for r in results)
    cited_total = sum(r.get("retrieved_sources_count", 0) for r in results)
    positives = [r for r in results if (r.get("expected_sources") or [])]
    route_miss_cases = [r for r in positives if r.get("route_miss_expected_law")]
    dominant_mismatch_cases = [r for r in positives if r.get("dominant_law_mismatch")]
    fallback_cases = [r for r in results if r.get("fallback_triggered")]
    fallback_recovered_cases = [r for r in fallback_cases if r.get("fallback_recovered")]

    failing_intent_counts: dict[str, int] = defaultdict(int)
    for row in route_miss_cases:
        for term in row.get("intent_terms") or []:
            failing_intent_counts[term] += 1
    failing_intent_clusters = sorted(
        (
            {"intent_term": term, "count": count}
            for term, count in failing_intent_counts.items()
        ),
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
        "route_miss_expected_law_rate": (len(route_miss_cases) / len(positives)) if positives else 0.0,
        "dominant_law_mismatch_count": len(dominant_mismatch_cases),
        "dominant_law_mismatch_rate": (len(dominant_mismatch_cases) / len(positives)) if positives else 0.0,
        "fallback_triggered_count": len(fallback_cases),
        "fallback_recovered_count": len(fallback_recovered_cases),
        "fallback_recovery_rate": (len(fallback_recovered_cases) / len(fallback_cases)) if fallback_cases else 0.0,
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

    fallback_rows = [row for row in results if (row.get("fallback_reason") or "").startswith("uncertainty")]
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
    parser = argparse.ArgumentParser(description="Analyze cross-law contamination in retrieval results.")
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
    logger.info(
        "Using trust profile: %s (version=%s)",
        resolved_profile,
        settings.trust_profile_version,
    )
    if args.retrieval_k_initial is not None and args.retrieval_k_initial > 0:
        settings.retrieval_k_initial = args.retrieval_k_initial
    chain = LegalRAGChain(settings=settings)

    questions = load_questions(args.questions)
    if args.sample_size > 0:
        questions = questions[: args.sample_size]

    logger.info("Analyzing %s questions (retrieval_k_initial=%s)", len(questions), settings.retrieval_k_initial)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(questions, start=1):
        rows.append(analyze_question(chain, row, retrieval_k_initial=settings.retrieval_k_initial))
        if idx % 10 == 0 or idx == len(questions):
            logger.info("Processed %s/%s questions", idx, len(questions))

    aggregate = summarize(rows)
    hard_cluster_summary = build_hard_cluster_summary(rows)
    report = {
        "trust_profile_name": settings.trust_profile_name,
        "trust_profile_version": settings.trust_profile_version,
        "aggregate": aggregate,
        "hard_cluster_summary": hard_cluster_summary,
        "per_question": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    logger.info("Saved contamination report to %s", args.output)
    logger.info(
        "Contamination rate=%.1f%%, singleton_foreign_rate=%.1f%%, unexpected_citation_rate=%.1f%%",
        aggregate["contamination_rate"] * 100.0,
        aggregate["singleton_foreign_rate"] * 100.0,
        aggregate["unexpected_citation_rate"] * 100.0,
    )


if __name__ == "__main__":
    # Keep local analysis runs out of LangSmith.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    main()
