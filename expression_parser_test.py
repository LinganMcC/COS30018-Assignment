import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_LABELS = HERE / "data" / "custom_samples" / "expressions" / "labels.csv"
DEFAULT_IMAGES = HERE / "data" / "custom_samples" / "expressions" / "ready"
# same path setup as gui.py: the backend lives in models/cnn
sys.path.insert(0, str(HERE / "models" / "cnn"))
import gui_backend as backend  # noqa: E402
from expression_parser import ExpressionError, evaluate, format_result  # noqa: E402

MODEL = "CNN + operators"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
TAG = re.compile(r"_s(\d+)w(\d+)", re.IGNORECASE)   # e001_s1w3 -> s=1, w=3


# Helpers
def normalise(text):
    """Map look-alike symbols to one form and drop spaces, so '10÷2×3' == '10/2*3'."""
    table = {"×": "*", "x": "*", "X": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}
    return "".join(table.get(c, c) for c in text if not c.isspace())


def bracket_group(expr):
    """Complexity group by bracket nesting depth (used when there is no level column)."""
    depth = deepest = 0
    for c in expr:
        if c == "(":
            depth += 1
            deepest = max(deepest, depth)
        elif c == ")":
            depth -= 1
    return {0: "no brackets", 1: "brackets"}.get(deepest, "nested brackets")


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / d, (centre + margin) / d


def same_value(a, b):
    """Equal as numbers; tolerates a true result that was rounded in the CSV (e.g. 0.333333)."""
    return a == b or abs(float(a) - float(b)) <= 1e-6


def find_images(folder, row):
    """All images for a row: <id>.ext or <id>_anything.ext, in the folder or folder/<tester>/."""
    stem = row["id"]
    found = []
    for base in (Path(folder), Path(folder) / row["tester"]):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.suffix.lower() in IMAGE_EXTS and (p.stem == stem or p.stem.startswith(stem + "_")):
                found.append(p)
        if found:
            break
    return sorted(found)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(DEFAULT_LABELS),
                    help="CSV with columns: id,expression (optional: tester,level,written_as,result) "
                         f"(default: {DEFAULT_LABELS.relative_to(HERE)})")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES),
                    help=f"folder containing the handwritten images (default: {DEFAULT_IMAGES.relative_to(HERE)})")
    ap.add_argument("--model-path", help="optional .keras file instead of the frozen CNN + operators model")
    ap.add_argument("--csv", help="write per-image results to this CSV")
    ap.add_argument("--save-failures", help="folder to copy failing input images into")
    args = ap.parse_args()

    if not Path(args.labels).is_file():
        sys.exit(f"Labels file not found: {args.labels}\n"
                 "Put it at that location or pass --labels path/to/labels.csv")
    if not Path(args.images).is_dir():
        sys.exit(f"Image folder not found: {args.images}\n"
                 "Put the images there or pass --images path/to/folder")

    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows or not {"id", "expression"} <= set(rows[0]):
        sys.exit("The labels CSV needs at least the columns: id,expression")
    for r in rows:
        for optional in ("tester", "level", "written_as", "result"):
            r.setdefault(optional, "")

    model = backend.load_model(MODEL, args.model_path)

    stats = defaultdict(lambda: defaultdict(int))
    records, failures, label_warnings, no_images = [], [], [], []
    if args.save_failures:
        os.makedirs(args.save_failures, exist_ok=True)

    for row in rows:
        rid, true_expr = row["id"], row["expression"]
        written = row["written_as"] or true_expr

        # Ground truth: the result column, else computed from the expression
        try:
            computed = evaluate(true_expr)
        except ExpressionError as e:
            print(f"SKIPPED {rid}: the true expression '{true_expr}' is not valid ({e})")
            continue
        if row["result"]:
            try:
                true_val = Fraction(row["result"])
            except (ValueError, ZeroDivisionError):
                print(f"SKIPPED {rid}: result '{row['result']}' is not a number")
                continue
            if not same_value(true_val, computed):
                label_warnings.append((rid, true_expr, row["result"], format_result(computed)))
        else:
            true_val = computed

        paths = find_images(args.images, row)
        if not paths:
            no_images.append(rid)
            continue

        for path in paths:
            img = Image.open(path)
            pipeline_error = None
            try:
                text = backend.recognise(model, img)["text"]
            except Exception as e:  # keep going; count it as a failure
                text, pipeline_error = "", str(e)
            try:
                val, err = evaluate(text), None
            except ExpressionError as e:
                val, err = None, str(e)

            result_ok = err is None and same_value(val, true_val)
            text_ok = normalise(text) == normalise(written)
            cause = ""
            if not result_ok:
                if pipeline_error:
                    cause = "pipeline error"
                elif len(normalise(text)) != len(normalise(written)):
                    cause = "segmentation (symbol count wrong)"
                else:
                    cause = "recognition (symbol misread)"

            level = row["level"] or bracket_group(true_expr)
            keys = ["ALL", f"level: {level}"]
            if row["tester"]:
                keys.append(f"tester: {row['tester']}")
            tag = TAG.search(path.stem)
            if tag:
                keys += [f"s: {tag.group(1)}", f"w: {tag.group(2)}"]
            for key in keys:
                s = stats[key]
                s["n"] += 1
                s["result_ok"] += result_ok
                s["text_ok"] += text_ok
                s["parse_fail"] += err is not None
                s[cause] += bool(cause)

            shown_result = format_result(val) if err is None else ""
            records.append({"image": path.name, "id": rid, "tester": row["tester"], "level": level,
                            "true_expression": true_expr, "written_as": written,
                            "true_result": format_result(true_val), "recognised": text,
                            "result": shown_result, "error": err or pipeline_error or "",
                            "result_correct": int(result_ok), "text_exact": int(text_ok), "cause": cause})
            if not result_ok:
                failures.append((path.name, level, written, text, shown_result or f"error: {err}", cause))
                if args.save_failures:
                    img.convert("RGB").save(Path(args.save_failures) / f"{path.stem}.png")

    # Report
    if no_images:
        print(f"\nNo image found for {len(no_images)} label row(s): {', '.join(no_images[:15])}"
              + (" ..." if len(no_images) > 15 else ""))
        print("  (check --images points at the right folder)")
    if label_warnings:
        print("\nWARNING: these rows have a result that disagrees with their expression "
              "(check the labels):")
        for rid, expr, given, computed in label_warnings:
            print(f"  {rid}: {expr}  result column = {given}, parser says {computed}")

    if not records:
        sys.exit("No test cases were run.")

    def table(title, keys):
        print(f"\n{title}")
        print(f"{'':22}{'n':>4}   {'end-to-end (95% CI)':26}{'exact text':>11}{'parse fails':>13}")
        for key in keys:
            s = stats[key]
            lo, hi = wilson(s["result_ok"], s["n"])
            label = key.split(": ", 1)[-1] if key != "ALL" else "ALL"
            line = f"{label:22}{s['n']:>4}   {s['result_ok'] / s['n']:>5.1%} ({lo:.0%}-{hi:.0%})"
            print(line.ljust(52) + f"{s['text_ok'] / s['n']:>11.1%}{s['parse_fail']:>13}")

    def keys_with(prefix):
        return sorted((k for k in stats if k.startswith(prefix)), key=lambda k: (len(k), k))

    print(f"\nModel: {MODEL}    images tested: {stats['ALL']['n']}")
    table("Overall", ["ALL"])
    table("By level (or bracket depth if there is no level column)", keys_with("level: "))
    if keys_with("tester: "):
        table("By tester", keys_with("tester: "))
    if keys_with("s: "):
        table("By 's' value in the file name (e001_s1w3 -> 1)", keys_with("s: "))
        table("By 'w' value in the file name (e001_s1w3 -> 3)", keys_with("w: "))

    causes = [c for c in ("segmentation (symbol count wrong)", "recognition (symbol misread)", "pipeline error")
              if stats["ALL"][c]]
    if causes:
        print("\nWhere the failures came from:")
        for c in causes:
            print(f"  {c}: {stats['ALL'][c]}")
    if failures:
        print("\nFailures (image | level | written | recognised | result | cause):")
        for name, level, written, text, val, cause in failures:
            print(f"  {name} | {level} | {written} | {text!r} | {val} | {cause}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        print(f"\nPer-image results written to {args.csv}")


if __name__ == "__main__":
    main()