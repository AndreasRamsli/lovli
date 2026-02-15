#!/usr/bin/env python3
"""
Editorial injection precision benchmark: A/B comparison with editorial ON vs OFF.

Quantifies how editorial note injection affects citation_precision, unexpected_citation_rate,
and non_editorial_clean_success. Supports offline runs via --cache.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from lovli.chain import LegalRAGChain
from lovli.config import get_settings
from lovli.eval_utils import validate_questions

# Load sweep module by path (scripts is not a package)
import importlib.util
_sweep_path = ROOT_DIR / "scripts" / "sweep_retrieval_thresholds.py"
_spec = importlib.util.spec_from_file_location("sweep_retrieval_thresholds", _sweep_path)
_sweep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep_mod)
apply_combo_to_chain = _sweep_mod.apply_combo_to_chain
load_questions = _sweep_mod.load_questions
precompute_candidates = _sweep_mod.precompute_candidates
_env_flag = _sweep_mod._env_flag

DEFAULT_CACHE_PATH = ROOT_DIR / "eval" / "editorial_precision_candidates.json"

# Default config values (match sweep defaults)
RETRIEVAL_K_INITIAL = 15
RETRIEVAL_K = 5
CONFIDENCE = 0.45
MIN_DOC_SCORE = 0.35
MIN_GAP = 0.05
TOP_SCORE_CEILING = 0.70


def _matches_expected_source(cited_source: dict, expected_source: dict) -> bool:
    """Check precise law-aware match with prefix-compatible article ID semantics."""
    cited_law = (cited_source.get("law_id") or "").strip()
    cited_article = (cited_source.get("article_id") or "").strip()
    expected_law = (expected_source.get("law_id") or "").strip()
    expected_article = (expected_source.get("article_id") or "").strip()
    if not expected_law or not expected_article:
        return False
    return cited_law == expected_law and cited_article.startswith(expected_article)


def _has_attached_editorial(provision_rows: list[dict], editorial_candidates: list[dict]) -> bool:
    """Return True when any kept provision has linked/chapter editorial candidates."""
    if not provision_rows or not editorial_candidates:
        return False
    provision_article_pairs = {
        ((row.get("law_id") or "").strip(), (row.get("article_id") or "").strip())
        for row in provision_rows
        if (row.get("law_id") or "").strip() and (row.get("article_id") or "").strip()
    }
    provision_chapter_pairs = {
        ((row.get("law_id") or "").strip(), (row.get("chapter_id") or "").strip())
        for row in provision_rows
        if (row.get("law_id") or "").strip() and (row.get("chapter_id") or "").strip()
    }
    for candidate in editorial_candidates:
        law_id = (candidate.get("law_id") or "").strip()
        linked = (candidate.get("linked_provision_id") or "").strip()
        chapter_id = (candidate.get("chapter_id") or "").strip()
        if law_id and linked and (law_id, linked) in provision_article_pairs:
            return True
        if law_id and chapter_id and (law_id, chapter_id) in provision_chapter_pairs:
            return True
    return False


def _compute_per_question_metrics(
    row: dict, min_sources: int, editorial_candidates: list
) -> dict:
    """
    Simulate retrieval for one question and return precision, unexpected_rate, cited_editorial,
    and budget metrics (provisions-only vs inflated).
    """
    expected = set(row.get("expected_articles", []))
    expected_sources = row.get("expected_sources", []) or []
    candidates = row.get("candidates", [])
    expects_editorial_context = bool(row.get("expects_editorial_context", False))

    subset = candidates[:RETRIEVAL_K_INITIAL]
    ranked = sorted(subset, key=lambda x: x["score"], reverse=True)[:RETRIEVAL_K]

    kept = [c for c in ranked if c["score"] >= MIN_DOC_SCORE]
    min_sources_val = min_sources
    if len(kept) < min(min_sources_val, len(ranked)):
        kept = ranked[: min(min_sources_val, len(ranked))]

    provision_kept = [c for c in kept if c.get("doc_type") != "editorial_note"]
    scores = [c["score"] for c in provision_kept]
    cited_ids = [c["article_id"] for c in provision_kept if c.get("article_id")]
    cited_sources = [
        {"law_id": c.get("law_id", ""), "article_id": c.get("article_id", "")}
        for c in provision_kept
        if c.get("article_id")
    ]
    cited_editorial = _has_attached_editorial(provision_kept, editorial_candidates)
    top_score = scores[0] if scores else None

    precision = 0.0
    unexpected_rate = 0.0

    if expected:
        if expected_sources:
            matched_citations = 0
            for cited in cited_sources:
                for exp in expected_sources:
                    if _matches_expected_source(cited, exp):
                        matched_citations += 1
                        break
            precision = (matched_citations / len(cited_sources)) if cited_sources else 0.0
            unexpected_rate = (
                (len(cited_sources) - matched_citations) / len(cited_sources)
                if cited_sources
                else 0.0
            )
        else:
            matched_citations = 0
            for cid in cited_ids:
                for exp in expected:
                    if cid.startswith(exp):
                        matched_citations += 1
                        break
            precision = (matched_citations / len(cited_ids)) if cited_ids else 0.0
            unexpected_rate = (
                (len(cited_ids) - matched_citations) / len(cited_ids)
                if cited_ids
                else 0.0
            )

    return {
        "precision": precision,
        "unexpected_rate": unexpected_rate,
        "cited_editorial": cited_editorial,
        "expects_editorial_context": expects_editorial_context,
        "top_score": top_score,
    }


def run_benchmark(chain_or_settings, cached_candidates: list[dict]) -> dict:
    """
    Run A/B comparison: editorial OFF vs editorial ON.
    Returns aggregates and per-question deltas.
    chain_or_settings: LegalRAGChain (live) or Settings (offline/cache mode).
    """
    settings_obj = getattr(chain_or_settings, "settings", chain_or_settings)
    min_sources = getattr(settings_obj, "reranker_min_sources", 2)
    if hasattr(chain_or_settings, "settings"):
        apply_combo_to_chain(
            chain_or_settings,
            retrieval_k_initial=RETRIEVAL_K_INITIAL,
            retrieval_k=RETRIEVAL_K,
            confidence=CONFIDENCE,
            min_doc=MIN_DOC_SCORE,
            min_gap=MIN_GAP,
            top_score_ceiling=TOP_SCORE_CEILING,
        )

    per_question_off: list[dict] = []
    per_question_on: list[dict] = []

    for row in cached_candidates:
        m_off = _compute_per_question_metrics(
            row, min_sources, editorial_candidates=[]
        )
        m_off["id"] = row.get("id")
        per_question_off.append(m_off)

        m_on = _compute_per_question_metrics(
            row, min_sources, editorial_candidates=row.get("editorial_candidates", [])
        )
        m_on["id"] = row.get("id")
        per_question_on.append(m_on)

    # Positive questions = those with expected_articles (contribute to precision/unexpected)
    positive_ids = {row.get("id") for row in cached_candidates if row.get("expected_articles")}
    pos_off = [r for r in per_question_off if r["id"] in positive_ids]
    pos_on = [r for r in per_question_on if r["id"] in positive_ids]

    def mean_vals(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    precision_off = mean_vals([r["precision"] for r in pos_off])
    precision_on = mean_vals([r["precision"] for r in pos_on])
    unexpected_off = mean_vals([r["unexpected_rate"] for r in pos_off])
    unexpected_on = mean_vals([r["unexpected_rate"] for r in pos_on])

    non_editorial_rows = [r for r in cached_candidates if not r.get("expects_editorial_context", False)]
    non_editorial_ids = {row.get("id") for row in non_editorial_rows}
    non_editorial_clean_off = sum(1 for r in per_question_off if r["id"] in non_editorial_ids and not r["cited_editorial"])
    non_editorial_clean_on = sum(1 for r in per_question_on if r["id"] in non_editorial_ids and not r["cited_editorial"])
    non_editorial_total = len(non_editorial_rows)

    # Per-question precision deltas (only positive questions)
    deltas = []
    for po, pn in zip(per_question_off, per_question_on):
        if po["id"] in positive_ids:
            deltas.append({
                "id": po["id"],
                "precision_off": po["precision"],
                "precision_on": pn["precision"],
                "delta": pn["precision"] - po["precision"],
                "expects_editorial_context": po["expects_editorial_context"],
            })
    deltas.sort(key=lambda x: x["delta"])

    return {
        "citation_precision_off": precision_off,
        "citation_precision_on": precision_on,
        "unexpected_citation_rate_off": unexpected_off,
        "unexpected_citation_rate_on": unexpected_on,
        "non_editorial_clean_off": non_editorial_clean_off,
        "non_editorial_clean_on": non_editorial_clean_on,
        "non_editorial_total": non_editorial_total,
        "deltas": deltas,
        "n_expect_editorial": sum(1 for r in cached_candidates if r.get("expects_editorial_context")),
        "n_positive": len(positive_ids),
        "n_negative": len(cached_candidates) - len(positive_ids),
    }


def print_report(result: dict) -> None:
    """Print markdown-formatted benchmark report."""
    r = result
    n = r["n_positive"] + r["n_negative"]
    print("=== Editorial Injection Precision Benchmark ===")
    print(f"Questions: {n} ({r['n_expect_editorial']} expect editorial, {r['n_positive']} positive, {r['n_negative']} negative)")
    print()
    print("                          | editorial OFF | editorial ON | delta")
    print("-" * 65)
    prec_off, prec_on = r["citation_precision_off"], r["citation_precision_on"]
    print(f"{'citation_precision':<25} | {prec_off:>13.3f} | {prec_on:>12.3f} | {prec_on - prec_off:+.3f}")
    unexp_off, unexp_on = r["unexpected_citation_rate_off"], r["unexpected_citation_rate_on"]
    print(f"{'unexpected_citation_rate':<25} | {unexp_off:>13.3f} | {unexp_on:>12.3f} | {unexp_on - unexp_off:+.3f}")
    nec_off, nec_on = r["non_editorial_clean_off"], r["non_editorial_clean_on"]
    nec_tot = r["non_editorial_total"]
    print(f"{'non_editorial_clean':<25} | {nec_off:>6}/{nec_tot:<6} | {nec_on:>6}/{nec_tot:<6} |")
    print()
    print("--- Worst precision regressions (editorial ON vs OFF) ---")
    for d in r["deltas"][:15]:
        exp_tag = "[expects_editorial=true]" if d["expects_editorial_context"] else ""
        print(f"  {d['id']}: {d['precision_off']:.3f} -> {d['precision_on']:.3f}  (delta {d['delta']:+.3f})  {exp_tag}")
    if len(r["deltas"]) > 15:
        print(f"  ... and {len(r['deltas']) - 15} more")


def main() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    parser = argparse.ArgumentParser(description="Editorial injection precision benchmark")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to load cached candidates from (enables fully offline run). "
        "If omitted, precomputes against live Qdrant and saves to eval/editorial_precision_candidates.json",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help=f"Path to save candidates cache (default: {DEFAULT_CACHE_PATH})",
    )
    args = parser.parse_args()

    if load_dotenv:
        load_dotenv(ROOT_DIR / ".env")

    questions_path = ROOT_DIR / "eval" / "questions.jsonl"
    questions = load_questions(questions_path)
    sample_size_raw = os.getenv("SWEEP_SAMPLE_SIZE")
    if sample_size_raw:
        sample_size = int(sample_size_raw)
        if sample_size > 0:
            questions = questions[:sample_size]
            print(f"Using sample size from SWEEP_SAMPLE_SIZE={sample_size}")

    if args.cache:
        with open(args.cache, "r", encoding="utf-8") as f:
            cached_candidates = json.load(f)
        print(f"Loaded {len(cached_candidates)} cached candidates from {args.cache}")
        settings = get_settings()
        result = run_benchmark(settings, cached_candidates)
    else:
        settings = get_settings()
        chain = LegalRAGChain(settings=settings)
        validate_questions(questions, chain, skip_index_scan=_env_flag("SWEEP_SKIP_INDEX_SCAN", True))
        print("Precomputing candidates against Qdrant...")
        cached_candidates = precompute_candidates(chain, questions, max_k_initial=RETRIEVAL_K_INITIAL)
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cached_candidates, f, ensure_ascii=False, indent=2)
        print(f"Saved candidates to {save_path}")
        result = run_benchmark(chain, cached_candidates)

    print_report(result)


if __name__ == "__main__":
    main()
