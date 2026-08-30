.PHONY: install lint test run train format clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest

train:
	python -m src.models.train

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
