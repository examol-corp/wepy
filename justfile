#!/usr/bin/env just --justfile

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

test:
    uv run pytest --import-mode=importlib tests/unit
