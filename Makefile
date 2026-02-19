.PHONY: install format test lint clean all help

help:
	@echo "Available commands:"
	@echo "  make install  - Install project dependencies"
	@echo "  make format   - Format code with ruff"
	@echo "  make test     - Run tests with pytest"
	@echo "  make lint     - Lint code with ruff"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make all      - Run format, lint, and test"

install:
	uv venv -p 3.11.0 .venv
	uv pip install -r pyproject.toml --all-extras
	uv lock

format:
	ruff format .

test:
	pytest tests/

lint:
	ruff check .

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

all: format lint test
