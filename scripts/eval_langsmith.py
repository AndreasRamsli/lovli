#!/usr/bin/env python3
"""
LangSmith evaluation script for Lovli RAG pipeline.
Evaluates retrieval relevance, citation accuracy, correctness, and groundedness.
"""

import logging
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
)

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

# Load environment variables from .env file
load_dotenv(root_dir / ".env")

from lovli.chain import LegalRAGChain
from lovli.config import Settings, get_settings
from lovli.eval_utils import infer_negative_type, validate_questions

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global chain instance (reused across evaluations)
_chain: LegalRAGChain | None = None


def get_langsmith_endpoint() -> str | None:
    """Get LangSmith endpoint from environment variables."""
    # Check both LANGSMITH_ENDPOINT and LANGCHAIN_ENDPOINT
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    return endpoint


def get_chain() -> LegalRAGChain:
    """Get or create the RAG chain instance."""
    global _chain
    if _chain is None:
        _chain = LegalRAGChain()
    return _chain


def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for evaluation.
    
    Uses chain.retrieve() for reranked sources, then generates answer.
    This ensures evaluation runs through the same pipeline as the app
    (including reranking and confidence gating).
    
    Args:
        inputs: Dictionary with 'question' key
        
    Returns:
        Dictionary with answer, cited article IDs, cited source tuples, and context.
    """
    chain = get_chain()
    question = inputs["question"]
    
    # Single retrieve call -- returns reranked sources with content
    sources, top_score, scores = chain.retrieve(question)
    
    # Extract cited article IDs
    cited_articles = [s.get("article_id", "unknown") for s in sources]
    cited_sources = [
        {
            "law_id": s.get("law_id", "unknown"),
            "article_id": s.get("article_id", "unknown"),
            "doc_type": s.get("doc_type", "provision"),
        }
        for s in sources
    ]
    
    # Format context as a list of document strings for evaluators
    context_documents = [
        f"Lov: {s.get('law_title', 'Unknown')} (§ {s.get('article_id', 'Unknown')})\n{s.get('content', '')}"
        for s in sources
    ]
    
    # Generate answer (with confidence gating)
    if chain.should_gate_answer(top_score, scores=scores):
        answer = "Jeg fant ikke et klart svar på spørsmålet ditt i lovtekstene. Kunne du prøve å omformulere spørsmålet eller være mer spesifikk?"
    elif not sources:
        answer = "Beklager, jeg kunne ikke finne informasjon om dette spørsmålet."
    else:
        # Generate answer using the same prompt as streaming
        context = chain._format_context(sources)
        messages = chain.prompt_template.format_messages(context=context, input=question)
        response = chain.llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
    
    return {
        "answer": answer,
        "cited_articles": cited_articles,
        "cited_sources": cited_sources,
        "retrieved_context": context_documents,
        "is_gated": chain.should_gate_answer(top_score, scores=scores),
        "reranker_top_score": top_score,
    }


def matches_expected(cited_id: str, expected_set: set) -> bool:
    """
    Check if a cited article ID matches any expected article using prefix matching.
    
    Args:
        cited_id: The cited article ID (e.g., "kapittel-3-paragraf-5-ledd-1")
        expected_set: Set of expected article IDs (e.g., {"kapittel-3-paragraf-5"})
        
    Returns:
        True if cited_id starts with any expected article ID
    """
    for exp in expected_set:
        if cited_id.startswith(exp):
            return True
    return False


def _matches_expected_source(cited_source: Dict[str, Any], expected_source: Dict[str, Any]) -> bool:
    """
    Check if cited source matches expected law + article pair.

    Article matching uses prefix semantics to preserve compatibility with
    paragraph/subsection citations.
    """
    cited_law = (cited_source.get("law_id") or "").strip()
    cited_article = (cited_source.get("article_id") or "").strip()
    expected_law = (expected_source.get("law_id") or "").strip()
    expected_article = (expected_source.get("article_id") or "").strip()
    if not expected_law or not expected_article:
        return False
    return cited_law == expected_law and cited_article.startswith(expected_article)


def create_citation_match_evaluator(settings: Settings):
    """
    Create a custom evaluator for citation matching.
    
    This evaluator checks if the cited articles match the expected articles
    using prefix matching logic.
    """
    def citation_match_evaluator(
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate citation match.
        
        Args:
            inputs: Input dictionary with 'question'
            outputs: Output dictionary with 'cited_articles' and optional 'cited_sources'
            reference_outputs: Reference dictionary with expected labels
            
        Returns:
            Evaluation result with score and comment
        """
        expected_sources = reference_outputs.get("expected_sources", []) or []
        expected_articles = set(reference_outputs.get("expected_articles", []))
        cited_articles = outputs.get("cited_articles", [])
        cited_sources = outputs.get("cited_sources", [])

        # Prefer precise law-aware matching when expected_sources are provided.
        if expected_sources:
            if not cited_sources:
                return {
                    "key": "citation_match",
                    "score": False,
                    "comment": "No sources cited. Expected law/article pairs were provided.",
                }

            matched_pairs = []
            found_expected_pairs = set()
            for idx, expected in enumerate(expected_sources):
                for cited in cited_sources:
                    if _matches_expected_source(cited, expected):
                        pair_key = (
                            expected.get("law_id", ""),
                            expected.get("article_id", ""),
                        )
                        found_expected_pairs.add(pair_key)
                        matched_pairs.append(
                            f"{cited.get('law_id', 'unknown')}:{cited.get('article_id', 'unknown')}"
                        )
                        break

            coverage = len(found_expected_pairs) / len(expected_sources)
            expected_fmt = ", ".join(
                f"{s.get('law_id', 'unknown')}:{s.get('article_id', 'unknown')}"
                for s in expected_sources
            )
            cited_fmt = ", ".join(
                f"{s.get('law_id', 'unknown')}:{s.get('article_id', 'unknown')}"
                for s in cited_sources
            )

            return {
                "key": "citation_match",
                "score": len(found_expected_pairs) > 0,
                "comment": (
                    "Law-aware citation matching. "
                    f"Coverage: {coverage:.1%} "
                    f"({len(found_expected_pairs)}/{len(expected_sources)} expected pairs). "
                    f"Matched: {', '.join(matched_pairs) if matched_pairs else 'none'}. "
                    f"Expected: {expected_fmt}. Cited: {cited_fmt}"
                ),
            }
        
        if not expected_articles:
            return {
                "key": "citation_match",
                "score": True,
                "comment": "No expected citation labels specified",
            }
        
        if not cited_articles:
            return {
                "key": "citation_match",
                "score": False,
                "comment": f"No articles cited. Expected: {', '.join(expected_articles)}",
            }
        
        # Check if any cited article matches an expected article
        matches = [
            cid for cid in cited_articles if matches_expected(cid, expected_articles)
        ]
        
        if matches:
            # Calculate coverage: fraction of expected articles that were cited
            found_expected = set()
            for cid in cited_articles:
                for exp in expected_articles:
                    if cid.startswith(exp):
                        found_expected.add(exp)
            coverage = len(found_expected) / len(expected_articles)
            
            comment = (
                f"Cited {len(matches)} matching article(s) out of {len(cited_articles)} cited. "
                f"Coverage: {coverage:.1%} ({len(found_expected)}/{len(expected_articles)} expected articles). "
                f"Expected: {', '.join(expected_articles)}, Cited: {', '.join(cited_articles)}"
            )
            
            return {
                "key": "citation_match",
                "score": True,
                "comment": comment,
            }
        else:
            return {
                "key": "citation_match",
                "score": False,
                "comment": (
                    f"No matching articles cited. "
                    f"Expected: {', '.join(expected_articles)}, "
                    f"Cited: {', '.join(cited_articles)}"
                ),
            }
    
    return citation_match_evaluator


def create_citation_precision_evaluator():
    """Measure citation precision so noisy extra citations are penalized."""

    def citation_precision_evaluator(
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_sources = reference_outputs.get("expected_sources", []) or []
        expected_articles = set(reference_outputs.get("expected_articles", []))
        cited_sources = outputs.get("cited_sources", []) or []
        cited_articles = outputs.get("cited_articles", []) or []

        # Skip rows without positive expectation.
        if not expected_sources and not expected_articles:
            return {
                "key": "citation_precision",
                "score": True,
                "comment": "Skipped (no expected citation labels for this row).",
            }

        if expected_sources:
            if not cited_sources:
                return {
                    "key": "citation_precision",
                    "score": 0.0,
                    "comment": "No cited sources; precision=0.0 for positive-labeled row.",
                }
            tp = 0
            for cited in cited_sources:
                if any(_matches_expected_source(cited, exp) for exp in expected_sources):
                    tp += 1
            precision = tp / len(cited_sources)
            return {
                "key": "citation_precision",
                "score": precision,
                "comment": f"Law-aware precision={precision:.1%} ({tp}/{len(cited_sources)} matched cited sources).",
            }

        if not cited_articles:
            return {
                "key": "citation_precision",
                "score": 0.0,
                "comment": "No cited articles; precision=0.0 for positive-labeled row.",
            }

        tp = sum(1 for cid in cited_articles if matches_expected(cid, expected_articles))
        precision = tp / len(cited_articles)
        return {
            "key": "citation_precision",
            "score": precision,
            "comment": f"Article precision={precision:.1%} ({tp}/{len(cited_articles)} matched cited articles).",
        }

    return citation_precision_evaluator


def create_offtopic_contamination_evaluator():
    """
    Evaluate whether ambiguous/off-topic prompts incorrectly return citations.

    For dataset rows with empty expected_articles, we expect either:
    - gated/no-results behavior, or
    - no citations.
    """

    def offtopic_contamination_evaluator(
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_articles = reference_outputs.get("expected_articles", [])
        if expected_articles:
            return {
                "key": "offtopic_contamination",
                "score": True,
                "comment": "Skipped (question has expected article labels).",
            }

        cited_articles = outputs.get("cited_articles", [])
        answer = outputs.get("answer", "")
        is_gated = outputs.get("is_gated", False)
        no_results = "Beklager, jeg kunne ikke finne informasjon" in answer
        negative_type = infer_negative_type(reference_outputs)

        ok = is_gated or no_results or len(cited_articles) == 0
        return {
            "key": "offtopic_contamination",
            "score": ok,
            "comment": (
                "No expected articles. "
                f"negative_type={negative_type}. "
                f"is_gated={is_gated}, cited={len(cited_articles)}, "
                f"no_results={no_results}"
            ),
        }

    return offtopic_contamination_evaluator


def create_editorial_context_evaluator():
    """
    Evaluate whether editorial context is included when explicitly expected.

    If expects_editorial_context=true in reference outputs, at least one cited
    source should be tagged as doc_type=editorial_note.
    """

    def editorial_context_evaluator(
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        expects_editorial = bool(reference_outputs.get("expects_editorial_context", False))
        if not expects_editorial:
            return {
                "key": "editorial_context",
                "score": True,
                "comment": "Skipped (editorial context not expected for this row).",
            }

        cited_sources = outputs.get("cited_sources", []) or []
        editorial = [
            s for s in cited_sources
            if (s.get("doc_type") or "").strip().lower() == "editorial_note"
        ]
        return {
            "key": "editorial_context",
            "score": len(editorial) > 0,
            "comment": (
                f"expects_editorial_context=true, editorial_sources_found={len(editorial)}"
            ),
        }

    return editorial_context_evaluator


def main():
    """Run LangSmith evaluation."""
    settings = get_settings()
    
    # Check for API key (try both uppercase and lowercase)
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("langsmith_api_key")
    if not api_key:
        logger.error(
            "LANGSMITH_API_KEY environment variable not set. "
            "Please set it in your .env file (as LANGSMITH_API_KEY) or export it."
        )
        sys.exit(1)
    
    # Initialize LangSmith client with EU endpoint if specified
    endpoint = get_langsmith_endpoint()
    client_kwargs = {}
    if endpoint:
        logger.info(f"Using LangSmith endpoint: {endpoint}")
        client_kwargs["api_url"] = endpoint
    else:
        logger.info("Using default LangSmith endpoint (set LANGSMITH_ENDPOINT env var for EU)")
    
    # Client will auto-detect API key from environment, but we can pass it explicitly
    client_kwargs["api_key"] = api_key
    client = Client(**client_kwargs)
    
    # Dataset name (should match the one created by upload_dataset.py)
    dataset_name = "lovli-eval-questions"

    # Validate local eval labels before running remote evaluation.
    questions_path = root_dir / "eval" / "questions.jsonl"
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    validate_questions(questions, get_chain())
    
    # Get judge model from settings or use default
    judge_model = getattr(settings, "eval_judge_model", "gpt-4o-mini")
    logger.info(f"Using judge model: {judge_model}")
    
    # Create evaluators
    evaluators = []
    
    # 1. Retrieval relevance evaluator
    retrieval_relevance_evaluator = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedback_key="retrieval_relevance",
        model=f"openai:{judge_model}",
    )
    
    def wrapped_retrieval_relevance(inputs, outputs, reference_outputs):
        """Wrap retrieval relevance evaluator to format context correctly."""
        context = {"documents": outputs.get("retrieved_context", [])}
        return retrieval_relevance_evaluator(
            inputs=inputs,
            context=context,
        )
    
    evaluators.append(wrapped_retrieval_relevance)
    
    # 2. Citation match evaluator (custom)
    citation_match_evaluator = create_citation_match_evaluator(settings)
    evaluators.append(citation_match_evaluator)
    
    # 3. Off-topic contamination evaluator (for expected_articles=[])
    evaluators.append(create_offtopic_contamination_evaluator())

    # 4. Citation precision evaluator (penalize noisy extra citations)
    evaluators.append(create_citation_precision_evaluator())

    # 5. Editorial context evaluator (optional per-row expectation)
    evaluators.append(create_editorial_context_evaluator())

    # 6. Correctness evaluator
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model=f"openai:{judge_model}",
    )
    
    def wrapped_correctness(inputs, outputs, reference_outputs):
        """Wrap correctness evaluator to use notes as reference."""
        # Use notes from reference_outputs as the reference answer
        reference_answer = reference_outputs.get("notes", "")
        return correctness_evaluator(
            inputs=inputs,
            outputs={"answer": outputs.get("answer", "")},
            reference_outputs={"answer": reference_answer},
        )
    
    evaluators.append(wrapped_correctness)
    
    # 7. Groundedness evaluator
    groundedness_evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        model=f"openai:{judge_model}",
    )
    
    def wrapped_groundedness(inputs, outputs, reference_outputs):
        """Wrap groundedness evaluator to format context correctly."""
        context = {"documents": outputs.get("retrieved_context", [])}
        return groundedness_evaluator(
            context=context,
            outputs={"answer": outputs.get("answer", "")},
        )
    
    evaluators.append(wrapped_groundedness)
    
    # Get project name from settings or use default
    project_name = getattr(settings, "langsmith_project", "lovli-evals")
    
    # Run evaluation
    logger.info(f"Starting evaluation on dataset '{dataset_name}'...")
    logger.info(f"Using project: {project_name}")
    
    try:
        experiment_results = client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix="lovli-eval",
            max_concurrency=2,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation completed!")
        logger.info("=" * 60)
        logger.info(f"Experiment: {experiment_results.experiment_name}")
        logger.info(f"View results at: {experiment_results.experiment_url}")
        logger.info("=" * 60)
        
        # Print summary statistics (if available)
        try:
            if hasattr(experiment_results, "results") and experiment_results.results:
                logger.info("\nSummary Statistics:")
                evaluator_names = [
                    "retrieval_relevance",
                    "citation_match",
                    "citation_precision",
                    "offtopic_contamination",
                    "editorial_context",
                    "correctness",
                    "groundedness",
                ]
                for evaluator_name in evaluator_names:
                    scores = []
                    for r in experiment_results.results:
                        if hasattr(r, "feedback_results") and r.feedback_results:
                            feedback = r.feedback_results.get(evaluator_name)
                            if feedback and isinstance(feedback, dict):
                                score = feedback.get("score")
                                if score is not None:
                                    # Convert boolean to float for averaging
                                    score_val = 1.0 if score is True else (0.0 if score is False else float(score))
                                    scores.append(score_val)
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        logger.info(f"  {evaluator_name}: {avg_score:.2%} average ({len(scores)}/{len(experiment_results.results)} evaluated)")

                # Off-topic breakdown by negative type (parsed from evaluator comment).
                neg_breakdown: Dict[str, list[float]] = {
                    "ambiguity": [],
                    "offtopic_legal": [],
                    "offtopic_nonlegal": [],
                    "unknown": [],
                }
                for r in experiment_results.results:
                    if not (hasattr(r, "feedback_results") and r.feedback_results):
                        continue
                    feedback = r.feedback_results.get("offtopic_contamination")
                    if not (feedback and isinstance(feedback, dict)):
                        continue
                    score = feedback.get("score")
                    comment = str(feedback.get("comment", ""))
                    match = re.search(r"negative_type=([a-z_]+)", comment)
                    neg_type = match.group(1) if match else "unknown"
                    if neg_type not in neg_breakdown:
                        neg_type = "unknown"
                    if score is not None:
                        neg_breakdown[neg_type].append(
                            1.0 if score is True else (0.0 if score is False else float(score))
                        )
                logger.info("  offtopic_contamination by type:")
                for neg_type, vals in neg_breakdown.items():
                    if vals:
                        logger.info("    %s: %.2f%% (%s)", neg_type, 100 * sum(vals) / len(vals), len(vals))
        except Exception as e:
            logger.debug(f"Could not compute summary statistics: {e}")
            logger.info("View detailed results in the LangSmith UI")
        
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
