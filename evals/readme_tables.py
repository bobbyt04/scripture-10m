"""The README's results tables, built from evals/results/*.json. tests/test_readme.py checks the README has them.

    python evals/readme_tables.py   # paste the output into the Results section of README.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evals" / "results"
BOOKS = ("gita", "quran", "bible")
NAMES = {"gita": "Gita", "quran": "Quran", "bible": "Bible"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def loss_table(r):
    loss = r["loss"]
    models = {m["name"]: m for m in loss["models"]}
    ours = models["Scripture-10M"]

    def ci(m, b):
        lo, hi = m[b]["ci95_nats"]
        return f"{lo:.3f}-{hi:.3f}"

    rows = ["| Book | Scripture-10M, nats (bits/char) | 95% interval | Best baseline | Baseline, nats (bits/char) "
            "| 95% interval |",
            "|---|---|---|---|---|---|"]
    for b in (*BOOKS, "overall"):
        base = models[loss["best_baseline"][b]]
        cells = [NAMES.get(b, "Mean of the three"), f"{ours[b]['nats']:.3f} ({ours[b]['bpc']:.3f})",
                 ci(ours, b) if b in BOOKS else "", base["name"], f"{base[b]['nats']:.3f} ({base[b]['bpc']:.3f})",
                 ci(base, b) if b in BOOKS else ""]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def memorisation_table(verse, book):
    rows = ["| Text | Gita | Quran | Bible |", "|---|---|---|---|"]

    def row(label, r, kind):
        cells = []
        for b in BOOKS:
            m = r["memorisation"][b][kind]
            cells.append(f"{m['copied_from_any']:.1%} (longest {m['longest_run']})")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    row(f"Generated, verse-sampled run (step {verse['step']})", verse, "generated")
    row(f"Generated, book-sampled run (step {book['step']})", book, "generated")
    row("Real held-out verses", verse, "heldout_real")
    return "\n".join(rows)


def style_table(r):
    style = r["style"]
    rows = ["| Texts | Gita | Quran | Bible |", "|---|---|---|---|"]
    for name, row in style["rows"].items():
        rows.append(f"| {name} | " + " | ".join(f"{row[b]['counts'][b]}/{row[b]['n']}" for b in BOOKS) + " |")
    noise = style["uniform_noise"]
    rows.append("| uniform noise (share sent to each book) | " +
                " | ".join(f"{noise['counts'][b]}/{noise['n']}" for b in BOOKS) + " |")
    return "\n".join(rows)


def wellformed_table(r):
    wf = r["wellformed"]
    return "\n".join([
        "| | Gita | Quran | Bible |", "|---|---|---|---|",
        "| Generated words found in the training text | " +
        " | ".join(f"{wf[b]['real_word_frac']:.1%}" for b in BOOKS) + " |",
        "| Mean length of generated verses, characters | " +
        " | ".join(f"{wf[b]['mean_verse_chars']:.0f}" for b in BOOKS) + " |",
        "| Mean length of real held-out verses | " +
        " | ".join(f"{wf[b]['real_mean_verse_chars']:.0f}" for b in BOOKS) + " |",
    ])


def tables():
    verse, book = load("model.json"), load("book_sampling.json")
    return [loss_table(verse), memorisation_table(verse, book), style_table(verse), wellformed_table(verse)]


if __name__ == "__main__":
    print("\n\n".join(tables()))
