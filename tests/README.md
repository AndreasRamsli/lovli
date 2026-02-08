# Lovli Test Suite

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lovli --cov-report=html

# Run specific test file
pytest tests/test_utils.py

# Run specific test
pytest tests/test_utils.py::test_extract_chat_history_empty

# Run only unit tests
pytest -m unit
```

## Test Structure

- `test_utils.py` - Tests for utility functions (chat history extraction)
- `test_chain.py` - Tests for LegalRAGChain (confidence gating, validation, reranking)
- `test_indexer.py` - Tests for sparse vector conversion logic

## Coverage Goals

- **Current**: Basic unit tests for critical functions
- **Target**: 80%+ coverage for core modules

## Adding New Tests

1. Create test file: `tests/test_<module>.py`
2. Import the module: `from lovli.<module> import <function>`
3. Write test functions: `def test_<function_name>():`
4. Use fixtures from `conftest.py` when available

## Test Categories

- **Unit tests** (`@pytest.mark.unit`): Fast, isolated tests
- **Integration tests** (`@pytest.mark.integration`): Tests requiring external services
- **Slow tests** (`@pytest.mark.slow`): Tests that take >1 second
