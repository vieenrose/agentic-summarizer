"""Entity-swap twins for en DECISIONS/ACTIONS (the inversion class v4 showed:
wrong names, numbers, polarity). COHERENT-swap design: find one entity that
appears in BOTH the chunk and the completion, replace the SAME old->new in both.
This teaches the model to extract the entity actually present (counterfactual
extraction) without ever creating an incoherent chunk/completion pair.
"""
import json
import random
import re
import sys

sys.path.insert(0, "src")

POLARITY = [("approved", "rejected"), ("agreed", "declined"), ("accepted", "rejected"),
            ("selected", "rejected"), ("confirmed", "cancelled"), ("adopted", "rejected")]

NAME_POOL = ["Alice", "Bob", "Carol", "David", "Erin", "Frank", "Grace", "Henry",
             "Iris", "Jack", "Kate", "Leo", "Mia", "Nate", "Owen", "Paul"]

#: words that are capitalized but are not person names (sentence leaders, verbs)
STOP = {"The", "This", "That", "They", "There", "These", "Provide", "Define", "Focus",
        "Create", "Calculate", "Discuss", "Use", "Target", "Phase", "Evaluate",
        "Exclude", "Report", "Review", "Confirm", "Send", "Seek", "Add", "Investigate",
        "Compare", "Warehouse", "Summary", "Decision", "Action", "Open", "Topic",
        "Industrial", "Marketing", "Grad", "Professor", "PhD", "User", "Interface"}


def _candidates(kind, comp, chunk, rng):
    """Yield (old, new) pairs where `old` appears in BOTH comp and chunk."""
    if kind == "number":
        for m in re.finditer(r"\b\d+(?:\.\d+)?\b", comp):
            # skip numbers inside anchors
            if any(a.start() <= m.start() < a.end() for a in re.finditer(r"\[[^\]]*\]", comp)):
                continue
            old = m.group(0)
            if old in chunk:
                new = str(rng.randint(2, 30))
                while new == old:
                    new = str(rng.randint(2, 30))
                yield old, new
    elif kind == "polarity":
        for a, b in POLARITY:
            for old, new in ((a, b), (b, a)):
                if old in comp.lower() and old in chunk.lower():
                    # case-preserving replace on both sides
                    yield old, new
    elif kind == "name":
        for n in sorted(set(re.findall(r"\b[A-Z][a-z]{2,}\b", comp)), key=len, reverse=True):
            if n in STOP:
                continue
            if n in chunk:
                new = rng.choice(NAME_POOL)
                if new.lower() != n.lower():
                    yield n, new


def _replace_case(text, old, new):
    """Replace old with new, preserving the original's capitalization."""
    m = re.search(re.escape(old), text, re.I)
    if not m:
        return None
    rep = new if m.group(0).islower() else (new.capitalize() if len(new) > 1 else new.upper())
    return text[:m.start()] + rep + text[m.end():]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/sft/minicpm-p15d.jsonl")
    ap.add_argument("--out", default="data/sft/entity-twins.jsonl")
    ap.add_argument("--n", type=int, default=160)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    en = [r for r in rows if r.get("lang") == "en" and
          r["completion"].strip().startswith(("ADD ACTIONS", "ADD DECISIONS"))]
    rng = random.Random(23)
    rng.shuffle(en)

    out = []
    for r in en:
        if len(out) >= args.n:
            break
        marker = "CHUNK:\n"
        if marker not in r["prompt"]:
            continue
        head, chunk = r["prompt"].split(marker, 1)
        comp = r["completion"]
        made = False
        for kind in ("number", "polarity", "name"):
            for old, new in _candidates(kind, comp, chunk, rng):
                c2 = _replace_case(comp, old, new)
                ch2 = _replace_case(chunk, old, new)
                if c2 is None or ch2 is None or c2 == comp or ch2 == chunk:
                    continue
                twin = dict(r)
                twin["completion"] = c2
                twin["prompt"] = head + marker + ch2
                twin["twin"] = True
                out.append(twin)
                made = True
                break
            if made:
                break

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} coherent entity-swap twins -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
