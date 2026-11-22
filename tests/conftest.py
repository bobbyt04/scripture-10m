import json
import random

import pytest

WORDS = {
    "gita": ["arjuna", "krishna", "self", "yoga", "action", "wise"],
    "quran": ["allah", "lord", "believers", "mercy", "messenger", "day"],
    "bible": ["god", "israel", "moses", "king", "jesus", "unto"],
}
SIZES = {"gita": 100, "quran": 200, "bible": 1200}  # unequal on purpose, like the real books


@pytest.fixture(scope="session")
def tiny_data_dir(tmp_path_factory):
    """A small stand-in for data/processed/: three books of made-up verses, each with its own words."""
    d = tmp_path_factory.mktemp("processed")
    rng = random.Random(0)
    for book, n in SIZES.items():
        with open(d / f"{book}.jsonl", "w") as f:
            for i in range(n):
                text = " ".join(rng.choice(WORDS[book]) for _ in range(rng.randint(4, 9))).capitalize() + "."
                f.write(json.dumps({"book": book, "ref": f"1:{i + 1}", "chapter": 1, "verse": i + 1,
                                    "section": "", "text": text}) + "\n")
    return d
