#!/usr/bin/env python3
"""
Retrieval evaluation script for Lovli RAG pipeline.
Calculates Recall@k and Precision@k for the retrieval step.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lovli.chain import LegalRAGChain
from lovli.config import Settings

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

def evaluate_retrieval(
    chain: LegalRAGChain, 
    questions: List[Dict[str, Any]], 
    k: int = 5
) -> List[Dict[str, Any]]:
    results = []
    
    logger.info(f"Starting retrieval evaluation on {len(questions)} questions (k={k})...")
    
    for q in questions:
        query = q["question"]
        expected_ids = set(q["expected_articles"])
        
        start_time = time.time()
        # Use the vectorstore directly for retrieval-only eval
        docs = chain.vectorstore.similarity_search(query, k=k)
        latency = time.time() - start_time
        
        retrieved_ids = []
        retrieved_meta = []
        
        for doc in docs:
            # Handle both object and dict metadata
            meta = doc.metadata if hasattr(doc, "metadata") else doc.get("metadata", {})
            # Extract article ID (format might vary, assuming "article_id" is the key)
            # Based on chain.py, it looks for "article_id" in metadata
            art_id = meta.get("article_id", "unknown")
            retrieved_ids.append(art_id)
            retrieved_meta.append({
                "article_id": art_id,
                "title": meta.get("title", ""),
                "law": meta.get("law_title", "")
            })
            
        # Calculate metrics
        # Use prefix matching: retrieved "kapittel-3-paragraf-5-ledd-1" matches expected "kapittel-3-paragraf-5"
        def matches_expected(retrieved_id: str, expected_set: set) -> bool:
            for exp in expected_set:
                if retrieved_id.startswith(exp):
                    return True
            return False
        
        # Hit: Is ANY expected article (or sub-article) found?
        hits = [rid for rid in retrieved_ids if matches_expected(rid, expected_ids)]
        hit_rate = 1.0 if hits else 0.0
        
        # Recall: What fraction of expected articles were found (at least one sub-article)?
        found_expected = set()
        for rid in retrieved_ids:
            for exp in expected_ids:
                if rid.startswith(exp):
                    found_expected.add(exp)
        recall = len(found_expected) / len(expected_ids) if expected_ids else 0.0
        
        # Precision: What fraction of retrieved docs were relevant?
        precision = len(hits) / k
        
        result = {
            "id": q["id"],
            "question": query,
            "expected_articles": list(expected_ids),
            "retrieved_articles": retrieved_ids,
            "retrieved_metadata": retrieved_meta,
            "hit": hit_rate > 0,
            "recall": recall,
            "precision": precision,
            "latency_ms": latency * 1000
        }
        results.append(result)
        
    return results

def main():
    # Setup paths
    root_dir = Path(__file__).parent.parent
    questions_path = root_dir / "eval" / "questions.jsonl"
    output_path = root_dir / "eval" / "results_retrieval.jsonl"
    
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)
        
    # Initialize chain (using existing settings/env)
    # Ensure we use the same settings as the app
    try:
        chain = LegalRAGChain()
        k = chain.settings.retrieval_k
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        sys.exit(1)
        
    # Load questions
    questions = load_questions(questions_path)
    
    # Run evaluation
    results = evaluate_retrieval(chain, questions, k=k)
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    # Calculate summary metrics
    avg_hit_rate = sum(r["hit"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    
    print("\n=== Retrieval Evaluation Summary ===")
    print(f"Total Questions: {len(results)}")
    print(f"Top-k: {k}")
    print(f"Hit Rate (Any expected found): {avg_hit_rate:.2%}")
    print(f"Mean Recall (Fraction of expected found): {avg_recall:.2%}")
    print(f"Mean Precision (Fraction of retrieved relevant): {avg_precision:.2%}")
    print(f"Avg Latency: {avg_latency:.1f} ms")
    print(f"Detailed results saved to: {output_path}")
    
    # Simple pass/fail check (can be used for CI)
    if avg_hit_rate < 0.80:
        logger.warning("Hit rate is below 80% target!")

if __name__ == "__main__":
    main()
