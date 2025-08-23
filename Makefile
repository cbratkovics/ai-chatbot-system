.PHONY: install test lint format type-check ci-local clean

# Install all dependencies with Poetry
install:
	poetry install --with dev

# Run tests
test:
	poetry run pytest tests/ -v

# Run tests with coverage
test-cov:
	poetry run pytest tests/ -v --cov=api --cov=app --cov-report=term-missing --cov-report=html

# Lint code
lint:
	poetry run flake8 api/ app/ tests/
	poetry run black --check api/ app/ tests/
	poetry run isort --check-only api/ app/ tests/

# Format code
format:
	poetry run black api/ app/ tests/
	poetry run isort api/ app/ tests/

# Type checking
type-check:
	poetry run mypy api/ app/ --ignore-missing-imports

# Run all CI checks locally
ci-local: lint type-check test

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf .mypy_cache
