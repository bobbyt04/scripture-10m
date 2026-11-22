"""Build the advice search and the full-corpus model from the three books.

    python search/build.py           # retrain the corpus model, then rebuild the site artefacts (about a minute)
    python search/build.py --site    # only rebuild the site artefacts (after editing the pool, themes, bridge or safety)

Two outputs:

1. search/model/: the full-corpus model (used offline, never to pick advice verses).
   - Word rules (search/lexicon.json). Stopwords are the words found in the
     largest share of verses, minus a hand-written list of content words,
     plus a hand-written list of fillers. Words that are always capitalised
     in the books (names such as Judas) are never shortened.
   - Vocabulary. Every word is reduced to a base form (Lexicon.normalise) and
     filler words are dropped. Each verse becomes a list of term ids.
   - Word vectors (PPMI + SVD). Count how often two terms appear within a few
     words of each other (inside one chapter), turn the counts into positive
     pointwise mutual information ("do these two words meet more often than
     chance?"), and compress that table with a truncated SVD. The Gita and the
     Quran are much smaller than the Bible, so their counts are up-weighted.
   - Verse vectors: a weighted average of the verse's word vectors (rare words
     count more), minus the direction shared by all verses, length 1.
2. search/artifacts/: everything the website loads (search/FORMAT.md):
   search.json (the reviewed pool with display text and BM25 terms, themes,
   bridge, word rules, safety lists, settings), pool_vectors.i8 (the pool
   verses' vectors, for similar()), similarities.json (hand-reviewed
   parallels) and fixtures.json (expected results for testing a port).

Re-running the build after the pool shrinks just works: everything is derived
from search/pool/*.json, and bridge words missing from the pool are dropped.
"""

import argparse
import hashlib
import json
import re
import sys
from array import array
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripture.search import BOOKS, Lexicon, Searcher, words  # noqa: E402
from search.corpus import MODEL, Corpus  # noqa: E402

DATA = ROOT / "data" / "processed"
OUT = ROOT / "search" / "artifacts"
POOL = ROOT / "search" / "pool"
SENSITIVE_SET = POOL / "sensitive_set.json"
LEXICON = ROOT / "search" / "lexicon.json"
BRIDGE = ROOT / "search" / "bridge.json"
SAFETY = ROOT / "search" / "safety.json"
THEMES = ROOT / "search" / "themes.json"

PARAMS = {
    "dim": 128,            # word and verse vector size
    "min_count": 3,        # terms seen fewer times get no vector
    "window": 5,           # co-occurrence window, in content words
    "book_balance": 0.5,   # up-weight smaller books by (bible tokens / book tokens) ** this
    "cds": 0.75,           # context-distribution smoothing for PPMI
    "svd_power": 0.5,      # word vector = U * S ** this
    "sif_a": 1e-3,         # word weight a / (a + p(word)) when averaging into verse vectors
    "similar_vector_weight": 0.5,  # corpus similar(): mix of vector cosine and TF-IDF cosine
}
# The advice scorer's settings, chosen on the development cases (evals/search_eval.py).
SEARCH_PARAMS = {
    "bm25_k1": 1.2,
    "bm25_b": 0.75,
    "bridge_weight": 0.8,          # a bridge word counts this much of a typed word
    "theme_weight": 1.0,           # score = theme_weight * theme match + lex_weight * BM25
    "lex_weight": 0.2,
    "min_score": 0.3,              # a book's verses below this are not shown
    "sensitive_theme_score": 0.5,  # theme score given to a sensitive topic's gentle themes
    "min_theme_verses": 3,         # a book offers a theme only with at least this many verses for it
}
DOC_FIELDS = ("text", "why", "themes")   # what each pool verse's BM25 document is made of


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_verses():
    verses = {}
    for book in BOOKS:
        with open(DATA / f"{book}.jsonl") as f:
            verses[book] = [json.loads(line) for line in f]
    return verses


def load_pool(directory=POOL):
    return {book: load_json(Path(directory) / f"{book}.json") for book in BOOKS}


def corpus_sha256(verses):
    """A checksum of the refs and texts the model was built from."""
    h = hashlib.sha256()
    for book in BOOKS:
        for v in verses[book]:
            h.update(f'{book}\t{v["ref"]}\t{v["text"]}\n'.encode())
    return h.hexdigest()


# ----- word rules -----

def derive_stopwords(verses, cfg):
    """Words in the largest share of verses (each book weighted equally), minus 'keep', plus 'add'."""
    share = Counter()
    for book in BOOKS:
        n = len(verses[book])
        df = Counter(w for v in verses[book] for w in set(words(v["text"])))
        for w, c in df.items():
            share[w] += c / n / len(BOOKS)
    top = [w for w, _ in sorted(share.items(), key=lambda x: (-x[1], x[0]))[:cfg["top_by_verse_share"]]]
    return sorted((set(top) - set(cfg["keep"])) | set(cfg["add"]))


def proper_names(verses):
    """Lower-cased words that are capitalised every time they occur (names such as Judas)."""
    caps, lower = set(), set()
    for book in BOOKS:
        for v in verses[book]:
            for w in re.findall(r"[A-Za-z]+", v["text"]):
                (caps if w[0].isupper() else lower).add(w.lower())
    return caps - lower


def build_lexicon(verses, cfg):
    return {
        "stopwords": derive_stopwords(verses, cfg["stopwords"]),
        "query_stopwords": sorted(cfg["query_stopwords"]),
        "irregular": cfg["irregular"],
        "suffix_rules": cfg["suffix_rules"],
        "not_inflected": sorted(set(cfg["not_inflected"]) | proper_names(verses)),
    }


def base_forms(raw_words, lex):
    """{word: term}: each word's base form among `raw_words`, or "" for a filler word."""
    known = set(raw_words)
    out = {}
    for w in sorted(known):
        t = lex.normalise(w, known)
        out[w] = "" if (w in lex.stopwords or t in lex.stopwords) else t
    return out


# ----- the full-corpus model -----

def build_vocab(verses, min_count, lex):
    """Map every corpus word to a term; order terms by frequency."""
    raw = Counter(w for book in BOOKS for v in verses[book] for w in words(v["text"]))
    base = base_forms(raw, lex)
    count = Counter()
    for w, c in raw.items():
        if base[w]:
            count[base[w]] += c
    terms = sorted(count, key=lambda t: (-count[t], t))
    term_id = {t: i for i, t in enumerate(terms)}
    corpus_words = sorted(raw)
    word_term = [term_id[base[w]] if base[w] else -1 for w in corpus_words]
    lookup = dict(zip(corpus_words, word_term))
    verse_terms = [torch.tensor([lookup[w] for w in words(v["text"]) if lookup[w] >= 0], dtype=torch.long)
                   for book in BOOKS for v in verses[book]]
    return {
        "terms": terms, "count": [count[t] for t in terms], "words": corpus_words, "word_term": word_term,
        "n_vec": sum(1 for t in terms if count[t] >= min_count), "verse_terms": verse_terms,
    }


def chapter_streams(verses, verse_terms, n_vec):
    """One stream of term ids per chapter (terms with vectors only), with its book."""
    streams = []
    i = 0
    for book in BOOKS:
        current, key = [], None
        for v in verses[book]:
            k = (v["section"], v["chapter"])
            if k != key and current:
                streams.append((book, torch.cat(current)))
                current = []
            key = k
            t = verse_terms[i]
            current.append(t[t < n_vec])
            i += 1
        if current:
            streams.append((book, torch.cat(current)))
    return streams


def book_weights(verses, verse_terms, power):
    sizes, i = {}, 0
    for book in BOOKS:
        n = len(verses[book])
        sizes[book] = sum(len(t) for t in verse_terms[i:i + n])
        i += n
    return {b: (sizes["bible"] / sizes[b]) ** power for b in BOOKS}


def cooccurrence(streams, weights, n_vec, window):
    """Symmetric counts of terms at most `window` positions apart, each pair weighted by its book."""
    counts = torch.zeros(n_vec * n_vec)
    for book, ids in streams:
        for d in range(1, window + 1):
            if len(ids) <= d:
                break
            a, b = ids[:-d], ids[d:]
            w = torch.full((len(a),), weights[book])
            counts.index_add_(0, a * n_vec + b, w)
            counts.index_add_(0, b * n_vec + a, w)
    return counts.view(n_vec, n_vec)


def ppmi(c, cds):
    """Positive PMI: max(0, log P(w, c) / (P(w) P(c))), with the context distribution smoothed by cds."""
    row = c.sum(1, keepdim=True)
    col = c.sum(0, keepdim=True) ** cds
    return (c * col.sum() / (row * col).clamp(min=1e-12) + 1e-12).log().clamp(min=0)


def ppmi_svd_vectors(streams, weights, n_vec, p):
    """Word vectors from a truncated SVD of the PPMI co-occurrence matrix."""
    m = ppmi(cooccurrence(streams, weights, n_vec, p["window"]), p["cds"])
    torch.manual_seed(0)
    u, s, _ = torch.svd_lowrank(m, q=min(p["dim"], n_vec), niter=6)
    return u * s ** p["svd_power"]


def verse_vectors(verse_terms, wv, counts, a):
    """SIF-weighted average of unit word vectors, minus the shared direction."""
    unit = wv / wv.norm(dim=1, keepdim=True).clamp(min=1e-9)
    p = torch.tensor(counts, dtype=torch.float32)
    sif = a / (a + p / p.sum())
    n_vec = len(wv)
    out = torch.zeros(len(verse_terms), wv.shape[1])
    for i, t in enumerate(verse_terms):
        t = t[t < n_vec]
        if len(t):
            out[i] = (sif[t].unsqueeze(1) * unit[t]).sum(0)
    _, _, v = torch.svd_lowrank(out, q=1, niter=6)
    principal = v[:, 0]
    out -= (out @ principal).unsqueeze(1) * principal
    return out / out.norm(dim=1, keepdim=True).clamp(min=1e-9), principal


def train(verses=None, params=None, lexicon=None):
    """The full-corpus model. Returns (model dict, word vectors, verse vectors, verse terms)."""
    p = {**PARAMS, **(params or {})}
    verses = verses or load_verses()
    lexicon = lexicon or load_json(LEXICON)
    lex_data = build_lexicon(verses, lexicon)
    vocab = build_vocab(verses, p["min_count"], Lexicon(lex_data))
    n_vec = vocab["n_vec"]
    streams = chapter_streams(verses, vocab["verse_terms"], n_vec)
    weights = book_weights(verses, vocab["verse_terms"], p["book_balance"])
    wv = ppmi_svd_vectors(streams, weights, n_vec, p)
    vv, _ = verse_vectors(vocab["verse_terms"], wv, vocab["count"], p["sif_a"])
    model = {
        "version": 3,
        "dim": p["dim"],
        "params": {"similar_vector_weight": p["similar_vector_weight"]},
        "lexicon": lex_data,
        "n_vec": n_vec,
        "words": vocab["words"],
        "word_term": vocab["word_term"],
        "terms": vocab["terms"],
        "books": {b: {"refs": [v["ref"] for v in verses[b]]} for b in BOOKS},
        "inputs": {"corpus_sha256": corpus_sha256(verses), "verses": {b: len(verses[b]) for b in BOOKS},
                   "lexicon_sha256": sha256(lexicon)},
    }
    return model, wv, vv, vocab["verse_terms"]


def quantise(m):
    """int8 per row: the row scaled so its largest entry is +-127 (only the direction is kept)."""
    scale = m.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
    return torch.round(m / scale * 127).to(torch.int8)


def write_corpus(model, vv, verse_terms, out=MODEL):
    """search/model/: model.json, verse_vectors.i8, verse_terms.u16, verse_offsets.u32 (little-endian, no header)."""
    assert sys.byteorder == "little" and len(model["terms"]) < 65536
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    (out / "verse_vectors.i8").write_bytes(array("b", quantise(vv).flatten().tolist()).tobytes())
    (out / "verse_terms.u16").write_bytes(array("H", torch.cat(verse_terms).tolist()).tobytes())
    offsets = torch.tensor([0] + [len(t) for t in verse_terms]).cumsum(0)
    (out / "verse_offsets.u32").write_bytes(array("I", offsets.tolist()).tobytes())


# ----- the site artefacts -----

def display_text(text, v):
    """The verse as the site shows it, from the pool entry's hand-checked display fields.

    "fixes" corrects typos in the source ([[wrong, right], ...]), "trim" drops a
    leading heading or conjunction ("A Psalm of David. ", "And "), "trim_end"
    trailing words ("Selah."). The first letter is capitalised and a trailing
    comma, semicolon, colon or dash becomes a full stop.
    """
    for wrong, right in v.get("fixes", []):
        if text.count(wrong) != 1:
            raise ValueError(f"fix {wrong!r} does not occur exactly once in {text[:60]!r}")
        text = text.replace(wrong, right)
    trim, trim_end = v.get("trim", ""), v.get("trim_end", "")
    if not text.startswith(trim) or not text.endswith(trim_end):
        raise ValueError(f"trim {trim!r} / trim_end {trim_end!r} do not fit {text[:60]!r}")
    text = text[len(trim):len(text) - len(trim_end)].strip()
    text = text[:1].upper() + text[1:]
    if text[-1:] in (",", ";", ":", "-"):
        text = text[:-1].rstrip() + "."
    return text


def offered_themes(pool, minimum):
    """{book: set of themes}: the themes a book has at least `minimum` verses for."""
    out = {}
    for book in BOOKS:
        counts = Counter(t for v in pool[book] for t in v["themes"])
        out[book] = {t for t, c in counts.items() if c >= minimum}
    return out


def build_site(pool, verses, lexicon, themes, bridge, safety, sensitive_set, params=None, fields=DOC_FIELDS):
    """The search.json dict, built from the pool and the hand-written lists.

    A verse keeps only the themes its book offers (at least min_theme_verses
    verses); a verse left with none is not published. Verses listed in
    `sensitive_set` are the only ones shown for sensitive topics. `fields`
    picks what each verse's BM25 document holds: "text" (the verse), "why"
    (its note) and "themes" (its themes' names); the benchmark's ablation uses it.
    """
    p = {**SEARCH_PARAMS, **(params or {})}
    theme_data = {k: v for k, v in themes.items() if not k.startswith("_")}
    texts = {b: {v["ref"]: v["text"] for v in verses[b]} for b in BOOKS}
    offered = offered_themes(pool, p["min_theme_verses"])
    entries, raw = {}, {}
    for book in BOOKS:
        entries[book] = []
        sensitive = {x["ref"] for x in sensitive_set[book]}
        unknown = sensitive - {v["ref"] for v in pool[book]}
        if unknown:
            raise ValueError(f"sensitive set verses not in the {book} pool: {sorted(unknown)}")
        for v in pool[book]:
            tags = [t for t in v["themes"] if t in offered[book]]
            if not tags:
                continue
            text = display_text(texts[book][v["ref"]], v)
            doc = (words(text) if "text" in fields else []) + (words(v["why"]) if "why" in fields else []) \
                + ([w for t in tags for w in words(theme_data[t]["name"])] if "themes" in fields else [])
            raw[(book, v["ref"])] = doc
            entries[book].append({"ref": v["ref"], "text": text, "themes": tags, "why": v["why"],
                                  "sensitive": v["ref"] in sensitive})
    lex = Lexicon(lexicon)
    vocabulary = base_forms({w for doc in raw.values() for w in doc}, lex)
    for book in BOOKS:
        for e in entries[book]:
            e["terms"] = [vocabulary[w] for w in raw[(book, e["ref"])] if vocabulary[w]]
    site_lexicon = {**lexicon, "not_inflected": sorted(set(lexicon["not_inflected"]) & set(vocabulary))}
    model = {
        "version": 1,
        "params": p,
        "lexicon": site_lexicon,
        "vocabulary": vocabulary,
        "bridge": {},
        "bridge_replace": [],
        "themes": theme_data,
        "safety": {k: v for k, v in safety.items() if not k.startswith("_")},
        "pool": entries,
    }
    probe = Searcher(model)                              # keep only bridge words that are pool terms
    entries_b = {}
    for key, targets in bridge.items():
        if key.startswith("_"):
            continue
        kept = [t for t in targets if probe.term_of(t)]
        if kept:
            entries_b[key] = kept
    model["bridge"] = entries_b
    model["bridge_replace"] = sorted(set(bridge.get("_replace", [])) & set(entries_b))
    return model


def site_inputs(verses, pool, lexicon, themes, bridge, safety, sensitive_set):
    return {"corpus_sha256": corpus_sha256(verses), "pool_sha256": sha256(pool), "lexicon_sha256": sha256(lexicon),
            "sensitive_set_sha256": sha256(sensitive_set),
            "themes_sha256": sha256(themes), "bridge_sha256": sha256(bridge), "safety_sha256": sha256(safety),
            "pool_verses": {b: len(pool[b]) for b in BOOKS}}


def pool_vectors(corpus, site):
    """int8 rows of the published pool verses' vectors (gita, quran, bible, each in pool order)."""
    rows = [corpus.books[b]["vectors"][corpus.books[b]["index"][v["ref"]]] for b in BOOKS for v in site["pool"][b]]
    return quantise(torch.stack(rows))


def write_site(model, vectors, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = {**model, "dim": vectors.shape[1]}
    (out / "search.json").write_text(json.dumps(model, separators=(",", ":"), ensure_ascii=False))
    (out / "pool_vectors.i8").write_bytes(array("b", vectors.flatten().tolist()).tobytes())


# ----- fixtures for a port -----

FIXTURE_QUERIES = [
    # general
    "I got laid off and I'm scared about money", "my girlfriend broke up with me", "I feel burnt out",
    "my dad has cancer", "I'm anxious about my exams", "I can't forgive my brother", "my mother passed away",
    "I feel like my life has no purpose", "I'm jealous of my friend's success", "I keep procrastinating",
    "I want to help the poor", "I'm grateful for my life", "I feel guilty about my past",
    "I want inner peace and a calm mind", "should I lie to my boss", "I'm drowning in debt",
    "I've been drinking way too much every night", "working 70 hour weeks and completely burned out",
    "I lose my temper way too easily", "my best friend just stopped talking to me", "I'm stressed out at work",
    "I'm getting married next month", "I'm afraid of dying", "nobody appreciates my work",
    "the Lord is my shepherd", "with hardship goeth ease", "God", "help", "I just got promoted", "I failed my exam",
    # sensitive
    "I'm gay", "I'm gay and my parents rejected me", "I lost the pregnancy last week", "I'm the only brown kid in my class",
    "I'm blind", "my friend is Christian and I'm Muslim", "I hate my body", "I'm from a different caste",
    # crisis
    "I want to die", "my husband hits me", "thinking of jumping off the roof", "I've been cutting again",
    "I want to kil myself", "i dont want to be here anymore", "I'm going to kill that guy", "sucidal thoughts",
    "my stepdad comes into my room at night", "I took a bunch of pills last night on purpose",
    # idioms (not crisis)
    "I'm killing it at work", "I could murder a pizza", "I'm dead tired", "I bombed my presentation",
    "my brother beat me at chess", "I need to cut back on sugar", "my dog was put down", "I don't want to die",
    # no match
    "xyzzy", "!!!", "", "the and of unto",
]
FIXTURE_THEMES = ["grief", "money_worry", "self_worth", "patience"]
FIXTURE_SIMILAR = [("bible", "Psalms 23:1"), ("quran", "94:5"), ("gita", "2:47")]
FIXTURE_WORDS = ["I can't sleep, I'm scared & he won't call", "Don’t", "im ok, dont worry", "my friend's dog",
                 "it's 70 hour weeks", "WON'T", "ma'am isn't here"]


def fixtures(s):
    """Expected outputs for a browser port's tests (search/FORMAT.md, "fixtures.json")."""
    def hits(rows):
        return [[h["ref"], h["score"]] for h in rows]
    cases = []
    for q in FIXTURE_QUERIES:
        r = s.search(q, k=3)
        cases.append({
            "query": q, "words": words(q), "category": r["category"], "topic": r["topic"], "no_match": r["no_match"],
            "themes": [[t["id"], t["score"]] for t in r["themes"]],
            "terms": [[t, w, src] for t, (w, src) in s.expand(q).items()],
            "results": {b: [[h["ref"], h["score"], h["matched"]["themes"], [m["word"] for m in h["matched"]["words"]]]
                            for h in r["results"][b]] for b in BOOKS},
        })
    return {
        "about": "Expected outputs of the committed artefacts, for testing a port. See search/FORMAT.md for every field.",
        "words": [[t, words(t)] for t in FIXTURE_WORDS],
        "search": cases,
        "by_theme": [{"theme": t, "results": {b: hits(v) for b, v in s.by_theme(t, k=3).items()}} for t in FIXTURE_THEMES],
        "similar": [{"book": b, "ref": r, "results": {o: hits(v) for o, v in s.similar(b, r, k=3).items()}}
                    for b, r in FIXTURE_SIMILAR],
    }


def build_artifacts(out=OUT, model_dir=MODEL):
    """Rebuild search/artifacts/ from the pool, the hand-written lists and search/model/."""
    from search.similarities import build_similarities
    verses = load_verses()
    pool = load_pool()
    corpus_model = load_json(Path(model_dir) / "model.json")
    lexicon, themes, bridge, safety, sens = (load_json(p) for p in (LEXICON, THEMES, BRIDGE, SAFETY, SENSITIVE_SET))
    model = build_site(pool, verses, corpus_model["lexicon"], themes, bridge, safety, sens)
    model["inputs"] = site_inputs(verses, pool, lexicon, themes, bridge, safety, sens)
    corpus = Corpus.load(model_dir)
    write_site(model, pool_vectors(corpus, model), out)
    s = Searcher.load(out)
    sims = build_similarities(corpus)
    (Path(out) / "similarities.json").write_text(json.dumps(sims, indent=1, ensure_ascii=False) + "\n")
    (Path(out) / "fixtures.json").write_text(json.dumps(fixtures(s), indent=1, ensure_ascii=False) + "\n")
    dropped = sum(len(v) for k, v in bridge.items() if not k.startswith("_")) - sum(len(v) for v in model["bridge"].values())
    print(f"pool: {model['inputs']['pool_verses']}; published: " + ", ".join(f"{b} {len(model['pool'][b])}" for b in BOOKS)
          + f"; bridge: {len(model['bridge'])} keys kept "
          f"({dropped} targets not in the pool dropped)")
    total = 0
    for f in sorted(Path(out).iterdir()):
        total += f.stat().st_size
        print(f"{f.name:>20}  {f.stat().st_size / 1e3:8.1f} kB")
    print(f"{'total':>20}  {total / 1e3:8.1f} kB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", action="store_true", help="reuse search/model/ and only rebuild search/artifacts/")
    args = ap.parse_args()
    torch.set_num_threads(3)
    if not args.site:
        model, _, vv, vt = train()
        write_corpus(model, vv, vt)
    build_artifacts()


if __name__ == "__main__":
    main()
