"""The results tables in README.md match evals/results/*.json (regenerate them with evals/readme_tables.py)."""
from pathlib import Path

from evals.metrics import strip_tags
from evals.readme_tables import ROOT, tables


def test_readme_tables_match_results():
    readme = (ROOT / "README.md").read_text()
    for table in tables():
        assert table in readme, f"README.md is out of date; run python evals/readme_tables.py\n{table}"


def test_readme_samples_match_sample_files():
    readme = (ROOT / "README.md").read_text()
    for book in ("gita", "quran", "bible"):
        text = Path(ROOT / "samples" / f"{book}.md").read_text()
        first = text.split("```text\n", 1)[1].split("\n", 1)[0]  # first verse of the first sample
        assert strip_tags(first) in readme
