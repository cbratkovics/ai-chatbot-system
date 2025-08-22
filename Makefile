.PHONY: help install dev-install test lint format run docker-build docker-up docker-down clean migrate docs

# Variables
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose -f infrastructure/docker/docker-compose.yml
BACKEND_DIR := backend
FRONTEND_DIR := frontend

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)AI Chatbot System - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo "  make lint        - Run code linters and formatters"
	@echo "  make format      - Auto-format code"
	@echo ""
	@echo "Testing & Validation:"
	@echo "  make test        - Run all tests with coverage"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-int    - Run integration tests only"
	@echo "  make test-e2e    - Run end-to-end tests only"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make validate    - Validate all performance claims"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  make docker      - Build and run with Docker"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up   - Start Docker services"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make docker-logs - View Docker logs"
	@echo ""
	@echo "Production:"
	@echo "  make prod        - Build for production"
	@echo "  make deploy      - Deploy to production"
	@echo ""
	@echo "Monitoring:"
	@echo "  make monitoring  - Start monitoring stack (Grafana, Prometheus)"
	@echo "  make logs        - Tail application logs"
	@echo "  make status      - Show service status"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean generated files and caches"
	@echo "  make reset       - Reset database and caches"
	@echo "  make backup      - Backup database"
	@echo "  make restore     - Restore database from backup"

# Complete initial setup
setup: install migrate seed
	@echo "✅ Setup complete! Run 'make dev' to start development"

# Install all dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "📦 Installing Node.js dependencies..."
	@if [ -d "frontend" ]; then \
		cd frontend && npm install; \
	fi
	@echo "📦 Installing benchmark tools..."
	pip install locust pytest pytest-cov black isort flake8 mypy
	@echo "✅ Dependencies installed"

# Database operations
migrate:
	@echo "🔄 Running database migrations..."
	python manage.py migrate
	@echo "✅ Migrations complete"

seed:
	@echo "🌱 Seeding database..."
	python manage.py seed --sample-data
	@echo "✅ Database seeded"

reset:
	@echo "⚠️  Resetting database and caches..."
	docker-compose down -v
	docker-compose up -d postgres redis
	sleep 5
	python manage.py migrate
	python manage.py seed --sample-data
	@echo "✅ Reset complete"

# Development commands
dev: docker-services
	@echo "🚀 Starting development environment..."
	python manage.py runserver --reload

serve:
	@echo "🚀 Starting API server..."
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

shell:
	@echo "🐚 Opening interactive shell..."
	python manage.py shell

# Testing commands
test:
	@echo "🧪 Running all tests with coverage..."
	python manage.py test --coverage --verbose

test-unit:
	@echo "🧪 Running unit tests..."
	python manage.py test --unit

test-int:
	@echo "🧪 Running integration tests..."
	python manage.py test --integration

test-e2e:
	@echo "🧪 Running end-to-end tests..."
	python manage.py test --e2e

coverage:
	@echo "📊 Generating coverage report..."
	pytest --cov=api --cov-report=html --cov-report=term
	@echo "Coverage report: tests/coverage_reports/html/index.html"

# Benchmarking and validation
benchmark:
	@echo "⚡ Running performance benchmarks..."
	python manage.py benchmark

benchmark-baseline:
	@echo "⚡ Setting new baseline..."
	python manage.py benchmark --save-baseline

validate:
	@echo "✔️  Validating all claims..."
	python scripts/validate_claims.py

validate-full:
	@echo "✔️  Running benchmarks and validating..."
	python scripts/validate_claims.py --run-benchmarks

# Code quality
lint:
	@echo "🔍 Running linters..."
	python manage.py lint --check

format:
	@echo "✨ Formatting code..."
	black api/ tests/ benchmarks/
	isort api/ tests/ benchmarks/
	@echo "✅ Code formatted"

typecheck:
	@echo "🔍 Type checking..."
	mypy api/

security:
	@echo "🔒 Running security checks..."
	pip install safety bandit
	safety check
	bandit -r api/

# Docker commands
docker: docker-build docker-up
	@echo "✅ Docker environment ready"

docker-build:
	@echo "🏗️  Building Docker images..."
	docker-compose build

docker-up:
	@echo "🚀 Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

docker-services:
	@echo "🚀 Starting required services..."
	docker-compose up -d postgres redis
	@echo "Waiting for services to be ready..."
	@sleep 5

# Monitoring
monitoring:
	@echo "📊 Starting monitoring stack..."
	docker-compose up -d prometheus grafana jaeger
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"

logs:
	@echo "📋 Tailing application logs..."
	tail -f logs/app.log

status:
	@echo "📊 Service status..."
	python manage.py status

metrics:
	@echo "📈 Viewing metrics..."
	curl -s http://localhost:9090/metrics | head -50

# Production
prod: test lint
	@echo "🏭 Building for production..."
	python manage.py build all
	@echo "✅ Production build complete"

deploy:
	@echo "🚀 Deploying to production..."
	python manage.py deploy --environment production

deploy-staging:
	@echo "🚀 Deploying to staging..."
	python manage.py deploy --environment staging --dry-run

# Utilities
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf tests/coverage_reports/*
	rm -rf benchmarks/results/*
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache .mypy_cache
	@echo "✅ Cleaned"

backup:
	@echo "💾 Backing up database..."
	@mkdir -p backups
	docker-compose exec postgres pg_dump -U chatbot_user chatbot_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup complete"

restore:
	@echo "♻️  Restoring database..."
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make restore FILE=backups/backup_YYYYMMDD_HHMMSS.sql"; \
		exit 1; \
	fi
	docker-compose exec -T postgres psql -U chatbot_user chatbot_db < $(FILE)
	@echo "✅ Restore complete"

# Quick commands for common workflows
quick-test: docker-services test
	@echo "✅ Quick test complete"

quick-benchmark: docker-services benchmark validate
	@echo "✅ Quick benchmark complete"

full-check: install lint test benchmark validate
	@echo "✅ Full check complete - ready for deployment"

# Environment setup
env:
	@echo "🔧 Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please update with your values"; \
	else \
		echo ".env file already exists"; \
	fi

# Version and info
version:
	@echo "AI Chatbot System v1.0.0"
	@python --version
	@docker --version
	@node --version 2>/dev/null || echo "Node.js not installed"

info:
	@echo "📊 Project Information"
	@echo "====================="
	@echo "Lines of Python code:"
	@find . -name "*.py" -not -path "./venv/*" | xargs wc -l | tail -1
	@echo ""
	@echo "Test files:"
	@find tests -name "test_*.py" | wc -l
	@echo ""
	@echo "Benchmark files:"
	@ls -la benchmarks/load_tests/ | wc -l
	@echo ""
	@echo "Docker containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}" | grep chatbot || echo "None running"

# CI/CD helpers
ci-test:
	@echo "🔄 Running CI tests..."
	pytest --cov=api --cov-report=xml --cov-report=term

ci-lint:
	@echo "🔄 Running CI linting..."
	black --check api/ tests/ benchmarks/
	isort --check-only api/ tests/ benchmarks/
	flake8 api/ tests/ benchmarks/
	mypy api/

ci-benchmark:
	@echo "🔄 Running CI benchmarks..."
	python benchmarks/run_benchmarks.py --compare-baseline

# Development shortcuts
d: dev
s: serve
t: test
b: benchmark
v: validate
l: lint
c: clean

.DEFAULT_GOAL := help