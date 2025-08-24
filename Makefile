.PHONY: help install test lint format clean docker-build docker-up docker-down

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with Poetry
	poetry install --with dev

test:  ## Run tests with coverage
	poetry run pytest tests/ -v --cov=api --cov-report=term-missing

lint:  ## Run linting checks
	poetry run black --check api/ tests/
	poetry run isort --check-only api/ tests/
	poetry run flake8 api/ tests/
	poetry run mypy api/

format:  ## Format code
	poetry run black api/ tests/
	poetry run isort api/ tests/

security:  ## Run security checks
	poetry run bandit -r api/
	poetry run safety check

clean:  ## Clean up cache and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/

docker-build:  ## Build Docker image
	docker build -t chatbot-platform:latest .

docker-up:  ## Start services with docker-compose
	docker-compose up -d

docker-down:  ## Stop services
	docker-compose down

migrate:  ## Run database migrations
	poetry run alembic upgrade head

serve:  ## Start development server
	poetry run python -m api.main
