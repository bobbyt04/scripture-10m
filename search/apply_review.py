"""Apply a reviewer's decisions to the verse pool (search/pool/*.json).

    python search/apply_review.py path/to/review.json [--name review-1]

A review file is {"gita": [item, ...], "quran": [...], "bible": [...]}. Each
item has a "ref", an "action" and a "reason":

    remove          take the verse out of the pool
    retag           drop the themes in "drop_themes" (a verse left with no theme leaves the pool)
    not_universal   the verse is no longer shown for sensitive topics
    fix_why         replace the note with "new_why"
    display         how the verse is shown: "trim" (leading words to drop), "trim_end"
                    (trailing words to drop), "fixes" ([[wrong, right], ...] for typos in the
                    source). Each is checked against the verse text; data/processed is never changed.

The review file is copied to search/pool/<name>.json as the record, and
search/pool/REVIEW.md gets a section listing what was applied and anything
skipped (for example a verse an earlier review already removed). Run
`python search/build.py --site` afterwards.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS  # noqa: E402

POOL = ROOT / "search" / "pool"
DATA = ROOT / "data" / "processed"


def load_texts():
    out = {}
    for book in BOOKS:
        with open(DATA / f"{book}.jsonl") as f:
            out[book] = {r["ref"]: r["text"] for r in map(json.loads, f)}
    return out


def save_pool(book, rows, directory=POOL):
    """One verse per line, so a diff shows each change."""
    lines = ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    (Path(directory) / f"{book}.json").write_text("[\n" + lines + "\n]\n")


def check_display(text, item):
    """Raise if a display change does not fit the verse."""
    if not text.startswith(item.get("trim", "")):
        raise ValueError(f"trim {item.get('trim')!r} is not the start of the verse")
    if not text.endswith(item.get("trim_end", "")):
        raise ValueError(f"trim_end {item.get('trim_end')!r} is not the end of the verse")
    for wrong, _ in item.get("fixes", []):
        if text.count(wrong) != 1:
            raise ValueError(f"fix {wrong!r} does not occur exactly once")


def apply(review, pool, texts):
    """Apply a review to pool ({book: [verse, ...]}) in place. Returns (applied, skipped) lists of (book, item, note)."""
    applied, skipped = [], []
    for book in BOOKS:
        rows = {v["ref"]: v for v in pool[book]}
        for item in review.get(book, []):
            v = rows.get(item["ref"])
            action = item["action"]
            if v is None:
                skipped.append((book, item, "not in the pool (already removed?)"))
                continue
            if action == "remove":
                del rows[item["ref"]]
                applied.append((book, item, "removed"))
            elif action == "retag":
                v["themes"] = [t for t in v["themes"] if t not in item["drop_themes"]]
                if v["themes"]:
                    applied.append((book, item, "dropped " + ", ".join(item["drop_themes"])))
                else:
                    del rows[item["ref"]]
                    applied.append((book, item, "no theme left: removed"))
            elif action == "not_universal":
                v["universal"] = False
                applied.append((book, item, "not universal"))
            elif action == "fix_why":
                v["why"] = item["new_why"]
                applied.append((book, item, "new note"))
            elif action == "display":
                check_display(texts[book][item["ref"]], item)
                changed = [k for k in ("trim", "trim_end", "fixes") if item.get(k)]
                for k in changed:
                    v[k] = item[k]
                (applied if changed else skipped).append(
                    (book, item, "display: " + ", ".join(changed) if changed else "display item with no change"))
            else:
                raise ValueError(f"unknown action {action!r} for {book} {item['ref']}")
        pool[book] = [v for v in pool[book] if v["ref"] in rows]
    return applied, skipped


def report(name, applied, skipped, pool):
    counts = Counter((b, i["action"]) for b, i, _ in applied)
    lines = [f"\n## {name}\n",
             "Applied: " + "; ".join(f"{b} {a} {counts[(b, a)]}" for b in BOOKS
                                     for a in ("remove", "retag", "not_universal", "fix_why", "display")
                                     if counts[(b, a)]) + ".",
             "Pool after: " + ", ".join(f"{b} {len(pool[b])}" for b in BOOKS) + " verses.\n"]
    lines += [f"- {b} {i['ref']}: {i['action']} ({note}). {i.get('reason', '')}".rstrip() for b, i, note in applied]
    if skipped:
        lines.append("\nNot applied:\n")
        lines += [f"- {b} {i['ref']}: {i['action']}: {note}. {i.get('reason', '')}".rstrip() for b, i, note in skipped]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("review")
    ap.add_argument("--name", help="record name (default: the review file's name)")
    args = ap.parse_args()
    src = Path(args.review)
    name = args.name or src.stem
    review = json.loads(src.read_text())
    pool = {b: json.loads((POOL / f"{b}.json").read_text()) for b in BOOKS}
    applied, skipped = apply(review, pool, load_texts())
    for book in BOOKS:
        save_pool(book, pool[book])
    record = POOL / f"{name}.json"
    if src.resolve() != record.resolve():
        shutil.copyfile(src, record)
    log = POOL / "REVIEW.md"
    head = "" if log.exists() else ("# Pool reviews\n\nEvery review applied to search/pool/ by search/apply_review.py, "
                                    "oldest first. The review files themselves are kept next to this one.\n")
    with open(log, "a") as f:
        f.write(head + report(name, applied, skipped, pool))
    print(f"applied {len(applied)}, skipped {len(skipped)}; pool: " + ", ".join(f"{b} {len(pool[b])}" for b in BOOKS))


if __name__ == "__main__":
    main()
