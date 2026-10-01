.PHONY: install lint test notebooks ci

install:
	python -m pip install --upgrade pip
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	if [ -f dev-requirements.txt ]; then pip install -r dev-requirements.txt; fi
	pip install pytest pytest-cov ruff nbformat nbclient

lint:
	ruff check src tests --select E9,F63,F7,F82

test:
	python -m pytest -q tests --maxfail=1 --disable-warnings --cov=src --cov-report=term-missing

notebooks:
	python - << 'PY'
import os
import nbformat

bad = []
for root, _, files in os.walk("."):
    if any(x in root for x in [".git", ".venv", "__pycache__", ".ipynb_checkpoints"]):
        continue
    for f in files:
        if f.endswith(".ipynb"):
            p = os.path.join(root, f)
            try:
                nbformat.read(p, as_version=4)
            except Exception as e:
                bad.append((p, str(e)))

if bad:
    for p, e in bad:
        print(f"FAILED NOTEBOOK PARSE: {p}\n  {e}")
    raise SystemExit(1)

print("Notebook parse check passed.")
PY

ci: lint test notebooks
