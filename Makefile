.PHONY: install lint format test train run clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

tformat:
	ruff format src/ tests/

test:
	pytest tests/ --cov=src --cov-report=term-missing

train:
	python -m src.models.train

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
