# QS — project tasks. Everything here runs offline.
#
# The env targets are the only ones that touch the network: `env` needs it once
# to install the locked requirements, `env-offline` needs only wheelhouse/.
# `make check` runs on the venv but needs no GPU and no network; `env-check`
# needs the GPU and the prefetched weights cache.

PY := python3
VENV ?= .venv
VPY := $(VENV)/bin/python

.PHONY: help render check test diagrams svg clean env env-offline env-check gui-check lint

help:
	@echo "make render      regenerate STATUS.md, the Russian report and the timeline"
	@echo "make check       validate project.yml, run tests, render every diagram, lint"
	@echo "make test        unit tests only"
	@echo "make diagrams    verify every Mermaid diagram parses and renders"
	@echo "make svg         write docs/diagrams/*.svg for export"
	@echo "make env         create $(VENV) and install the locked requirements (online, once)"
	@echo "make env-offline the same, from wheelhouse/ only — make env-offline VENV=<path> for a fresh venv"
	@echo "make env-check   the env-marked tests — GPU and weights cache, offline"
	@echo "make gui-check   the gui-marked tests — needs a display"
	@echo "make lint        ruff and mypy on the venv"

render:
	@$(PY) tools/render.py

check:
	@echo "== project state =="
	@$(VPY) tools/render.py --check
	@echo "== tests =="
	@$(VPY) -m pytest tests/ -q
	@echo "== diagrams =="
	@$(VPY) tools/check_diagrams.py
	@echo "== lint =="
	@$(MAKE) lint

test:
	@$(PY) -m pytest tests/ -q

diagrams:
	@$(PY) tools/check_diagrams.py

svg:
	@$(PY) tools/check_diagrams.py --svg

env:
	@test -x $(VPY) || $(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --require-hashes -r requirements.txt -r requirements-dev.txt
	$(VENV)/bin/pip check

env-offline:
	@test -x $(VPY) || $(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --no-index --find-links wheelhouse/ \
		--require-hashes -r requirements.txt -r requirements-dev.txt
	$(VENV)/bin/pip check

env-check:
	HF_HOME=.hf TORCH_HOME=.torch HF_HUB_OFFLINE=1 $(VENV)/bin/pytest -m env

gui-check:
	$(VENV)/bin/pytest -m gui

lint:
	$(VENV)/bin/ruff check src tools tests
	$(VENV)/bin/mypy

clean:
	@rm -rf docs/diagrams/*.svg .pytest_cache
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
