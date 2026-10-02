from src.live_review import review_generated_text

def test_live_review_flags_short():
    res = review_generated_text("short")
    assert res["ok"] is False
    assert "short" in res["issues"][0].lower()
