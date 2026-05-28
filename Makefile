.PHONY: install test lint run-app clean

install:
	uv sync --all-extras
	cp -n .env.example .env || true

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	uv run ruff check src/ app/ tests/
	uv run ruff format --check src/ app/ tests/

format:
	uv run ruff format src/ app/ tests/

run-app:
	uv run streamlit run app/streamlit_app.py

run-notebook:
	uv run jupyter lab notebooks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/

download-data:
	uv run python src/data/ingestion.py

build-features:
	uv run python src/features/pipeline.py

train-model:
	uv run python src/models/train.py
