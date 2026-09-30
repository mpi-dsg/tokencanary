# Analysis targets need no API keys: they read only the committed results.
PY := .venv/bin/python

.PHONY: help setup report audit bounds control all

help:
	@echo "setup    create the venv and install pinned dependencies"
	@echo "report   per-cell statistics for every block"
	@echo "audit    independent integrity audit of the dataset"
	@echo "bounds   simultaneous upper bounds on inflation"
	@echo "control  positive control: detection rates and attack surface"
	@echo "all      report, audit, bounds, control"

setup:
	uv venv --python 3.13 .venv
	uv pip install --python $(PY) -r requirements.txt

report:
	$(PY) audit/final_report.py

audit:
	$(PY) audit/audit_integrity.py

bounds:
	$(PY) audit/equivalence_bounds.py

control:
	$(PY) audit/positive_control.py

all: report audit bounds control
