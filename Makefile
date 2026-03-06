SHELL := /bin/bash
RUNNER := uv run

setup-hooks:
	$(RUNNER) pre-commit install

check-hooks:
	$(RUNNER) pre-commit run --all-files
