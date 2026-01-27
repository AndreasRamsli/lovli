#!/usr/bin/env python3
"""
Answer evaluation script for Lovli RAG pipeline.
Evaluates citation accuracy and generates a review file for manual adjudication.
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lovli.chain import LegalRAGChain

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_questions(path: Path) -> List[Dict[str, Any]]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

def evaluate_answers(
    chain: LegalRAGChain, 
    questions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    results = []
    
    logger.info(f"Starting answer evaluation on {len(questions)} questions...")
    
    for q in questions:
        query = q["question"]
        expected_ids = set(q["expected_articles"])
        
        start_time = time.time()
        try:
            response = chain.query(query)
            latency = time.time() - start_time
            
            answer_text = response.get("answer", "")
            sources = response.get("sources", [])
            
            # Extract article IDs from sources
            cited_ids = []
            for source in sources:
                # Assuming source is a dict with 'article_id'
                if isinstance(source, dict):
                    cited_ids.append(source.get("article_id", "unknown"))
                else:
                    # Fallback if source structure changes
                    cited_ids.append(str(source))
            
            # Citation Metrics
            # Use prefix matching: cited "kapittel-3-paragraf-5-ledd-1" matches expected "kapittel-3-paragraf-5"
            def matches_expected(cited_id: str, expected_set: set) -> bool:
                for exp in expected_set:
                    if cited_id.startswith(exp):
                        return True
                return False
            
            # Match: Did we cite ANY of the expected articles (or sub-articles)?
            matches = [cid for cid in cited_ids if matches_expected(cid, expected_ids)]
            citation_match = len(matches) > 0
            
            # Coverage: What fraction of expected articles were cited (at least one sub-article)?
            found_expected = set()
            for cid in cited_ids:
                for exp in expected_ids:
                    if cid.startswith(exp):
                        found_expected.add(exp)
            citation_coverage = len(found_expected) / len(expected_ids) if expected_ids else 0.0
            
            # Hallucination check (proxy): Did we cite articles NOT in expected?
            # (Note: This is a weak proxy, as other articles might be relevant too)
            unexpected_citations = [cid for cid in cited_ids if not matches_expected(cid, expected_ids)]
            
            result = {
                "id": q["id"],
                "question": query,
                "expected_articles": list(expected_ids),
                "answer": answer_text,
                "cited_articles": cited_ids,
                "citation_match": citation_match,
                "citation_coverage": citation_coverage,
                "unexpected_citations": unexpected_citations,
                "latency_ms": latency * 1000
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing question {q['id']}: {e}")
            results.append({
                "id": q["id"],
                "question": query,
                "error": str(e),
                "citation_match": False,
                "citation_coverage": 0.0
            })
            
    return results

def generate_review_csv(results: List[Dict[str, Any]], output_path: Path):
    """Generate a CSV for manual review of failed/suspicious cases."""
    
    # Define "suspicious" as:
    # 1. No citation match (missed the point?)
    # 2. Unexpected citations (hallucination or just extra context?)
    # 3. Low coverage (< 1.0)
    
    review_cases = []
    for res in results:
        if "error" in res:
            continue
            
        status = "OK"
        if not res["citation_match"]:
            status = "MISSING_CITATION"
        elif res["citation_coverage"] < 1.0:
            status = "PARTIAL_CITATION"
        elif res["unexpected_citations"]:
            status = "EXTRA_CITATION"
            
        # We include all for review, but sort by status priority
        review_cases.append({
            "id": res["id"],
            "status": status,
            "question": res["question"],
            "answer": res["answer"][:200] + "..." if len(res["answer"]) > 200 else res["answer"],
            "expected": ", ".join(res["expected_articles"]),
            "cited": ", ".join(res["cited_articles"]),
            "correct_citation (y/n)": "",
            "correct_answer (y/n)": "",
            "comments": ""
        })
        
    # Sort: MISSING -> PARTIAL -> EXTRA -> OK
    priority = {"MISSING_CITATION": 0, "PARTIAL_CITATION": 1, "EXTRA_CITATION": 2, "OK": 3}
    review_cases.sort(key=lambda x: priority.get(x["status"], 4))
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_cases[0].keys())
        writer.writeheader()
        writer.writerows(review_cases)
        
    logger.info(f"Generated review CSV at {output_path} with {len(review_cases)} rows.")

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent
    questions_path = root_dir / "eval" / "questions.jsonl"
    output_path = root_dir / "eval" / "results_answer.jsonl"
    review_path = root_dir / "eval" / "review.csv"
    
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)
        
    # Initialize chain
    try:
        chain = LegalRAGChain()
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        sys.exit(1)
        
    # Load questions
    questions = load_questions(questions_path)
    
    # Run evaluation
    results = evaluate_answers(chain, questions)
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    # Generate review CSV
    generate_review_csv(results, review_path)
            
    # Calculate summary metrics
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("No valid results to summarize.")
        return

    avg_match_rate = sum(r["citation_match"] for r in valid_results) / len(valid_results)
    avg_coverage = sum(r["citation_coverage"] for r in valid_results) / len(valid_results)
    avg_latency = sum(r["latency_ms"] for r in valid_results) / len(valid_results)
    
    print("\n=== Answer & Citation Evaluation Summary ===")
    print(f"Total Questions: {len(results)}")
    print(f"Citation Match Rate (Any expected cited): {avg_match_rate:.2%}")
    print(f"Mean Citation Coverage (Fraction of expected cited): {avg_coverage:.2%}")
    print(f"Avg Latency: {avg_latency:.1f} ms")
    print(f"Detailed results saved to: {output_path}")
    print(f"Manual review file saved to: {review_path}")
    
    # Simple pass/fail check
    if avg_match_rate < 0.80:
        logger.warning("Citation match rate is below 80% target!")

if __name__ == "__main__":
    main()
