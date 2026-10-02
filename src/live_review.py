def review_generated_text(text: str) -> dict:
    issues = []
    if len(text.strip()) < 20:
        issues.append("Output too short.")
    if "TODO" in text:
        issues.append("Contains TODO placeholders.")
    return {"ok": len(issues) == 0, "issues": issues}
