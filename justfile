#!/usr/bin/env just --justfile

default_python := "3.13"


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
    uv run --all-extras --python {{python}} pytest tests/unit

test-comprehensive:
    uv run --all-extras --python 3.11 pytest tests/unit
    uv run --all-extras --python 3.12 pytest tests/unit
    uv run --all-extras --python 3.13 pytest tests/unit
    uv run --all-extras --python 3.14 pytest tests/unit
    

test-integration python=default_python:
    uv run --all-extras --python {{ python }} \
        pytest \
            --durations=0 \
            -s \
            -o log_cli=true --log-cli-level=INFO \
            tests/integration

clean:
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
