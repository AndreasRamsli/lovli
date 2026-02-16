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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load jsonl questions."""
    questions: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _matches_expected_source(cited_source: dict[str, str], expected_source: dict[str, str]) -> bool:
    """Match law_id exactly and article_id by prefix for compatibility."""
    cited_law = (cited_source.get("law_id") or "").strip()
    cited_article = (cited_source.get("article_id") or "").strip()
    expected_law = (expected_source.get("law_id") or "").strip()
    expected_article = (expected_source.get("article_id") or "").strip()
    if not expected_law or not expected_article:
        return False
    return cited_law == expected_law and cited_article.startswith(expected_article)


def analyze_question(chain: LegalRAGChain, row: dict[str, Any], retrieval_k_initial: int) -> dict[str, Any]:
    """Analyze one question and return law-level contamination diagnostics."""
    question = row["question"]
    expected_sources = row.get("expected_sources", []) or []
    expected_articles = row.get("expected_articles", []) or []

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
            if not any(_matches_expected_source(cited, exp) for exp in expected_sources):
                unexpected_sources.append(cited)
        matched_expected_pairs = sum(
            1 for exp in expected_sources if any(_matches_expected_source(cited, exp) for cited in cited_sources)
        )
    else:
        # For negatives, all citations are unexpected.
        unexpected_sources = cited_sources

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
    }


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
    report = {"aggregate": aggregate, "per_question": rows}
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
