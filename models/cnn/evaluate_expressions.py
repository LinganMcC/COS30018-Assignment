"""Extension test: read the team's handwritten expressions and score them.
Data: data/custom_samples/expressions/ready/  (e###_s#w#.jpg + labels.csv)
Outputs (models/experiments/eval/):
    expressions_report.md     tables for the report
    expressions_results.csv   one row per image: truth, read, result, correct
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from inference import REPO_ROOT, Recogniser

DATA = REPO_ROOT / "data" / "custom_samples" / "expressions" / "ready"
OUT_DIR = REPO_ROOT / "models" / "experiments" / "eval"
SHOW = {"*": "×", "/": "÷", "-": "−"}
LEVELS = {"1": "one operator", "2": "two operators", "3": "brackets", "4": "nested brackets"}


def lev(a: str, b: str) -> int:
    """Edit distance between two strings."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(expr: str) -> Optional[float]:
    """Value of a + - * / ( ) expression, or None if it is not well formed.

    Small recursive-descent evaluator used only for scoring (no eval()).
    Russell's parser can replace it.
    """
    tokens, i = [], 0
    while i < len(expr):
        if expr[i].isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(int(expr[i:j]))
            i = j
        else:
            tokens.append(expr[i])
            i += 1
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def factor():
        t = peek()
        if isinstance(t, int):
            return float(take())
        if t == "(":
            take()
            v = expression()
            if take() != ")":
                raise ValueError
            return v
        raise ValueError

    def term():
        v = factor()
        while peek() in ("*", "/"):
            op, r = take(), factor()
            v = v * r if op == "*" else v / r
        return v

    def expression():
        v = term()
        while peek() in ("+", "-"):
            op, r = take(), term()
            v = v + r if op == "+" else v - r
        return v

    try:
        v = expression()
        return v if pos == len(tokens) else None
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
        return None


def score(truth: str, read: str, true_result: float) -> dict:
    value = evaluate(read)
    return {
        "exact": read == truth,
        "edits": lev(read, truth),
        "length_ok": len(read) == len(truth),
        "read_result": "" if value is None else f"{value:g}",
        "result_ok": value is not None and abs(value - true_result) < 1e-9,
    }


def summarise(rows) -> dict:
    n = len(rows)
    chars = sum(len(r["truth"]) for r in rows)
    return {
        "n": n,
        "exact": sum(r["exact"] for r in rows) / n,
        "symbol_acc": max(0.0, 1 - sum(r["edits"] for r in rows) / chars),
        "length_ok": sum(r["length_ok"] for r in rows) / n,
        "result_ok": sum(r["result_ok"] for r in rows) / n,
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def symbol_stats(rows):
    """Per-class accuracy and confusions, on rows where the length matched."""
    total, ok, conf = Counter(), Counter(), Counter()
    for r in rows:
        if not r["length_ok"]:
            continue
        for t, p in zip(r["truth"], r["read"]):
            total[t] += 1
            ok[t] += t == p
            if t != p:
                conf[(t, p)] += 1
    return total, ok, conf


def report(model: str, rows) -> str:
    overall = summarise(rows)
    total, ok, conf = symbol_stats(rows)

    def row(name, s):
        return (f"| {name} | {s['n']} | {pct(s['exact'])} | {pct(s['symbol_acc'])} | "
                f"{pct(s['length_ok'])} | {pct(s['result_ok'])} |")

    md = [f"# Handwritten expression test – `{model}`", "",
          f"{overall['n']} cropped expressions from the team's sheets "
          "(`data/custom_samples/expressions/ready/`).", "",
          "| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |",
          "|---|---|---|---|---|---|", row("**All**", overall)]
    for lv in sorted({r["level"] for r in rows}):
        md.append(row(f"Level {lv} – {LEVELS.get(lv, lv)}", summarise([r for r in rows if r["level"] == lv])))
    md += ["", "Right length = the number of symbols found equals the truth (mostly a segmentation "
           "measure). Symbol accuracy = 1 − edit distance / true length.", "",
           "## Per symbol (expressions with the right length)", "",
           "| Symbol | Seen | Accuracy |", "|---|---|---|"]
    for s in sorted(total, key=lambda c: (not c.isdigit(), c)):
        md.append(f"| {SHOW.get(s, s)} | {total[s]} | {pct(ok[s] / total[s])} |")
    md += ["", "## Most frequent confusions", "", "| True | Read as | Count |", "|---|---|---|"]
    for (t, p), c in conf.most_common(10):
        md.append(f"| {SHOW.get(t, t)} | {SHOW.get(p, p)} | {c} |")
    md += ["", "## Per writer", "", "| Writer | Images | Exact | Symbol accuracy | Right length |",
           "|---|---|---|---|---|"]
    for wr in sorted({r["writer"] for r in rows}):
        s = summarise([r for r in rows if r["writer"] == wr])
        md.append(f"| {wr} | {s['n']} | {pct(s['exact'])} | {pct(s['symbol_acc'])} | {pct(s['length_ok'])} |")
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the handwritten expression test set.")
    ap.add_argument("--model", default="ext_best")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images")
    ap.add_argument("--no-soften", action="store_true", help="force the input blur off (ablation)")
    ap.add_argument("--soften", action="store_true", help="force the input blur on (ablation)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--sheets", default="3,4",
                    help="sheets to score; default 3,4 = writers never used in training "
                         "(ext_finetune.py trains on sheets 1-2). 'all' = every photo")
    args = ap.parse_args(argv)

    import cv2
    label_file = args.data / "labels.csv"
    if not label_file.exists():
        sys.exit(f"{label_file} not found - extract ready.zip into data/custom_samples/expressions/")
    labels = list(csv.DictReader(open(label_file, encoding="utf-8")))
    if args.sheets and args.sheets != "all":
        keep = set(args.sheets.split(","))
        labels = [l for l in labels if l["sheet"] in keep]
    labels = labels[:args.limit or None]
    rec = Recogniser(args.model, soften=False if args.no_soften else (True if args.soften else None))
    if not rec.reads_operators():
        sys.exit(f"{args.model} has no operator classes - use ext_best.")

    rows = []
    for k, lab in enumerate(labels, 1):
        img = cv2.imread(str(args.data / lab["file"]))
        read = rec.read_image(img)[0] if img is not None else ""
        rows.append({"file": lab["file"], "id": lab["id"], "level": lab["level"],
                     "writer": f"s{lab['sheet']}w{lab['writer']}", "truth": lab["expression"],
                     "read": read, "true_result": lab["result"],
                     **score(lab["expression"], read, float(lab["result"]))})
        if k % 100 == 0:
            print(f"  {k}/{len(labels)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["file", "id", "level", "writer", "truth", "read", "true_result", "read_result",
            "exact", "length_ok", "result_ok", "edits"]
    with open(OUT_DIR / f"expressions_results{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / f"expressions_report{args.tag}.md").write_text(
        report(args.model + (f", sheets {args.sheets}" if args.sheets != "all" else ""), rows),
        encoding="utf-8")

    s = summarise(rows)
    print(f"\n{s['n']} images | exact {pct(s['exact'])} | symbols {pct(s['symbol_acc'])} | "
          f"right length {pct(s['length_ok'])} | result {pct(s['result_ok'])}")
    print(f"-> {OUT_DIR / f'expressions_report{args.tag}.md'}")


if __name__ == "__main__":
    main()
