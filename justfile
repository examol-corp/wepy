#!/usr/bin/env just --justfile

default_python := "3.14"


fmt-check:
    uv run black --check src tests sphinx/conf.py

fmt:
    uv run black src tests sphinx/conf.py

fix-check:
    uv run isort --check src tests sphinx/conf.py
    uv run ruff check src tests sphinx/conf.py

fix:
    uv run isort src tests sphinx/conf.py
    uv run ruff check --fix src tests sphinx/conf.py

check:
    uv run mypy src

test python=default_python:
    uv sync --python {{python}} --all-extras
    uv run --python {{python}} pytest tests/unit

test-integration:
    uv run pytest --durations=0 -s -o log_cli=true --log-cli-level=INFO tests/integration

clean:
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
