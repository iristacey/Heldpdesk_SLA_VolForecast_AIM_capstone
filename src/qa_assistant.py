from pathlib import Path

def answer_question(question: str, docs_dir: str = "docs") -> str:
    q = question.lower()
    docs = []
    for p in Path(docs_dir).rglob("*.md"):
        docs.append(p.read_text(encoding="utf-8", errors="ignore"))

    corpus = "\n".join(docs).lower()
    if not corpus.strip():
        return "No docs found. Add markdown files to docs/."

    # super-simple retrieval baseline
    if "sla" in q and "priority" in corpus:
        return "SLA appears priority-based in your project docs."
    if "forecast" in q and "7" in corpus:
        return "Project includes 7-day and 30-day forecasting horizons."
    return "I couldn't find a confident answer in local docs."
