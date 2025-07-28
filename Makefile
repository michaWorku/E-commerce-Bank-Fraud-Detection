# Makefile for common development tasks

.PHONY: all setup clean test lint docs build deploy

all: setup lint test

setup:  ## Setup the project environment
	@echo "Setting up virtual environment and installing dependencies..."
	@if [ ! -d "venv" ]; then python3 -m venv venv; fi
	@source venv/bin/activate && pip install --upgrade pip
	@source venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete."

clean:  ## Clean up build artifacts and temporary files
	@echo "Cleaning up..."
	@find . -name "__pycache__" -exec rm -rf {} +
	@find . -name "*.pyc" -exec rm -f {} +
	@find . -name "*.egg-info" -exec rm -rf {} +
	@rm -rf build/ dist/ .pytest_cache/ htmlcov/ .mypy_cache/
	@echo "Clean complete."

test:  ## Run all tests
	@echo "Running tests..."
	@source venv/bin/activate && pytest
	@echo "Tests complete."

lint:  ## Run linters (flake8)
	@echo "Running linting..."
	@source venv/bin/activate && flake8 src/ tests/ scripts/
	@echo "Linting complete."

docs:  ## Build documentation
	@echo "Building documentation (requires Sphinx and make in docs/)..."
	@cd docs && make html
	@echo "Documentation built to docs/_build/html."

build:  ## Build the Python package
	@echo "Building package..."
	@source venv/bin/activate && python3 -m build
	@echo "Package built to dist/."

deploy:  ## Deploy the application (placeholder)
	@echo "Deploying application..."
	@echo "Deployment logic goes here."

help:  ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "[36m%-20s[0m %s\n", $$1, $$2}'

