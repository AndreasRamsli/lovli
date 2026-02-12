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
import sys
from pathlib import Path

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
                        "score": score,
                    }
                )

        cached.append(
            {
                "id": row.get("id"),
                "expected_articles": row.get("expected_articles", []),
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
) -> dict:
    """Compute retrieval metrics for one config combo using cached candidates."""
    retrieval_hits = 0
    retrieval_total = 0
    ambiguity_total = 0
    ambiguity_clean = 0
    top_scores: list[float] = []
    final_k = chain.settings.retrieval_k
    min_sources = chain.settings.reranker_min_sources

    for row in cached_candidates:
        expected = set(row.get("expected_articles", []))
        candidates = row.get("candidates", [])

        # Simulate retrieval_k_initial by truncating pre-rerank candidates.
        subset = candidates[:retrieval_k_initial]
        ranked = sorted(subset, key=lambda x: x["score"], reverse=True)[:final_k]

        # Simulate per-doc score filtering + floor.
        kept = [c for c in ranked if c["score"] >= min_doc_score]
        if len(kept) < min(min_sources, len(ranked)):
            kept = ranked[: min(min_sources, len(ranked))]

        scores = [c["score"] for c in kept]
        cited_ids = [c["article_id"] for c in kept if c.get("article_id")]
        top_score = scores[0] if scores else None

        if top_score is not None:
            top_scores.append(top_score)

        if expected:
            retrieval_total += 1
            if any(matches_expected(cid, expected) for cid in cited_ids):
                retrieval_hits += 1
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
                and scores[0] <= chain.settings.reranker_ambiguity_top_score_ceiling
            ):
                if (scores[0] - scores[1]) < chain.settings.reranker_ambiguity_min_gap:
                    is_gated = True
            if is_gated or not cited_ids:
                ambiguity_clean += 1

    recall_at_k = retrieval_hits / retrieval_total if retrieval_total else 0.0
    ambiguity_success = ambiguity_clean / ambiguity_total if ambiguity_total else 1.0
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return {
        "recall_at_k": recall_at_k,
        "ambiguity_success": ambiguity_success,
        "avg_top_score": avg_top_score,
    }


def apply_combo_to_chain(
    chain: LegalRAGChain,
    retrieval_k_initial: int,
    confidence: float,
    min_doc: float,
) -> None:
    """Apply sweep parameters to an existing chain instance."""
    chain.settings.retrieval_k_initial = retrieval_k_initial
    chain.settings.reranker_confidence_threshold = confidence
    chain.settings.reranker_min_doc_score = min_doc


def main() -> None:
    # Keep local sweeps from consuming LangSmith quota.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    if load_dotenv:
        load_dotenv(ROOT_DIR / ".env")
    questions_path = ROOT_DIR / "eval" / "questions.jsonl"
    questions = load_questions(questions_path)
    settings = get_settings()

    retrieval_k_initial_values = [15, 20, 25]
    confidence_values = [0.30, 0.40, 0.45]
    min_doc_values = [0.20, 0.30, 0.35]

    logger.info("Loaded %s questions", len(questions))
    logger.info("Starting retrieval sweep...")
    chain = LegalRAGChain(settings=settings)
    max_k_initial = max(retrieval_k_initial_values)
    logger.info("Precomputing candidates once with k=%s...", max_k_initial)
    cached_candidates = precompute_candidates(chain, questions, max_k_initial=max_k_initial)

    rows: list[dict] = []
    for retrieval_k_initial, confidence, min_doc in itertools.product(
        retrieval_k_initial_values,
        confidence_values,
        min_doc_values,
    ):
        apply_combo_to_chain(chain, retrieval_k_initial, confidence, min_doc)
        metrics = evaluate_combo(
            chain,
            cached_candidates,
            retrieval_k_initial=retrieval_k_initial,
            confidence_threshold=confidence,
            min_doc_score=min_doc,
        )
        row = {
            "retrieval_k_initial": retrieval_k_initial,
            "reranker_confidence_threshold": confidence,
            "reranker_min_doc_score": min_doc,
            **metrics,
        }
        rows.append(row)
        logger.info(
            "k_init=%s conf=%.2f min_doc=%.2f -> recall@k=%.3f ambiguity=%.3f avg_top=%.3f",
            retrieval_k_initial,
            confidence,
            min_doc,
            metrics["recall_at_k"],
            metrics["ambiguity_success"],
            metrics["avg_top_score"],
        )

    # Sort with ambiguity quality first, then recall.
    rows.sort(key=lambda r: (r["ambiguity_success"], r["recall_at_k"]), reverse=True)
    out_path = ROOT_DIR / "eval" / "retrieval_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("Saved results: %s", out_path)
    logger.info("Top 5 configurations:")
    for row in rows[:5]:
        logger.info(row)


if __name__ == "__main__":
    main()
