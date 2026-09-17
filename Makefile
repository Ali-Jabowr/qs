# QS — project tasks. Everything here runs offline.

PY := python3

.PHONY: help render check test diagrams svg clean

help:
	@echo "make render    regenerate STATUS.md, the Russian report and the timeline"
	@echo "make check     validate project.yml, run tests, render every diagram"
	@echo "make test      unit tests only"
	@echo "make diagrams  verify every Mermaid diagram parses and renders"
	@echo "make svg       write docs/diagrams/*.svg for export"

render:
	@$(PY) tools/render.py

check:
	@echo "== project state =="
	@$(PY) tools/render.py --check
	@echo "== tests =="
	@$(PY) -m pytest tests/ -q
	@echo "== diagrams =="
	@$(PY) tools/check_diagrams.py

test:
	@$(PY) -m pytest tests/ -q

diagrams:
	@$(PY) tools/check_diagrams.py

svg:
	@$(PY) tools/check_diagrams.py --svg

clean:
	@rm -rf docs/diagrams/*.svg .pytest_cache
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
