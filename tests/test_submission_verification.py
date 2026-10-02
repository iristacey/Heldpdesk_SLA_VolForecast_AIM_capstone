from pathlib import Path

def test_required_structure_exists():
    assert Path("README.md").exists(), "README.md is required"
    assert Path(".github/workflows/ci.yml").exists(), "CI workflow is required"
    assert Path("src").exists(), "src/ folder is required"
