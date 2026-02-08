#!/usr/bin/env python3
"""
Upload evaluation questions to LangSmith as a dataset.
This script only needs to be run once to create the dataset.
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Load environment variables from .env file
load_dotenv(root_dir / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_langsmith_endpoint() -> str | None:
    """Get LangSmith endpoint from environment variables."""
    # Check both LANGSMITH_ENDPOINT and LANGCHAIN_ENDPOINT
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        logger.debug(f"Found endpoint in env: {endpoint}")
    else:
        logger.debug("No endpoint found in LANGSMITH_ENDPOINT or LANGCHAIN_ENDPOINT")
    return endpoint


def main():
    """Upload questions.jsonl to LangSmith as a dataset."""
    questions_path = root_dir / "eval" / "questions.jsonl"
    
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)
    
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
    client_kwargs = {"api_key": api_key}
    if endpoint:
        logger.info(f"Using LangSmith endpoint: {endpoint}")
        client_kwargs["api_url"] = endpoint
    else:
        logger.warning("Using default LangSmith endpoint (set LANGSMITH_ENDPOINT env var for EU)")
        logger.info("Available env vars: " + ", ".join([k for k in os.environ.keys() if "LANGSMITH" in k or "LANGCHAIN" in k]))
    
    logger.debug(f"Client kwargs: {list(client_kwargs.keys())}")
    try:
        client = Client(**client_kwargs)
        # Test connection by getting API info
        logger.debug("Testing LangSmith connection...")
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith client: {e}")
        logger.error("Please verify:")
        logger.error("  1. LANGSMITH_API_KEY is correct")
        logger.error("  2. LANGSMITH_ENDPOINT is set correctly (https://eu.api.smith.langchain.com for EU)")
        logger.error("  3. Your API key has the necessary permissions")
        sys.exit(1)
    
    # Load questions from JSONL
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    logger.info(f"Loaded {len(questions)} questions from {questions_path}")
    
    # Create dataset
    dataset_name = "lovli-eval-questions"
    dataset_description = "Evaluation questions for Lovli RAG pipeline with expected articles and reference notes"
    
    try:
        # Check if dataset already exists
        existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
        if existing_datasets:
            logger.warning(f"Dataset '{dataset_name}' already exists. Use the existing dataset or delete it first.")
            dataset_id = existing_datasets[0].id
            logger.info(f"Existing dataset ID: {dataset_id}")
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description=dataset_description,
            )
            dataset_id = dataset.id
            logger.info(f"Created dataset '{dataset_name}' with ID: {dataset_id}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error creating dataset: {error_msg}")
        if "403" in error_msg or "Forbidden" in error_msg:
            logger.error("\n403 Forbidden error suggests:")
            logger.error("  1. Your API key may not have dataset creation permissions")
            logger.error("  2. Check your LangSmith workspace settings")
            logger.error("  3. Verify you're using the correct endpoint for your region")
            logger.error(f"  4. Current endpoint: {endpoint or 'default (api.smith.langchain.com)'}")
        elif "401" in error_msg or "Unauthorized" in error_msg:
            logger.error("\n401 Unauthorized error suggests:")
            logger.error("  1. Your API key is invalid or expired")
            logger.error("  2. Check that LANGSMITH_API_KEY is set correctly")
        sys.exit(1)
    
    # Prepare examples for LangSmith
    examples = []
    for q in questions:
        example = {
            "inputs": {
                "question": q["question"],
            },
            "outputs": {
                "expected_articles": q["expected_articles"],
                "notes": q.get("notes", ""),
            },
            "metadata": {
                "id": q.get("id", ""),
            },
        }
        examples.append(example)
    
    # Upload examples to dataset
    try:
        client.create_examples(
            dataset_id=dataset_id,
            examples=examples,
        )
        logger.info(f"Successfully uploaded {len(examples)} examples to dataset '{dataset_name}'")
        logger.info(f"Dataset URL: https://eu.smith.langchain.com/datasets/{dataset_id}")
    except Exception as e:
        logger.error(f"Error uploading examples: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
