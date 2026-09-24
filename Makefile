install:
	uv sync

all: run

run:
	uv run python -m src

debug:
	uv run python pdg -m src

clean:
	rm -rf .mypy_cache .DS_Store
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install all run debug clean lint lint-strict