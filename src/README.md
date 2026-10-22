# `src/` and `experiments/` – preprocessing, segmentation and test data

Owner: **John** (Task 1 preprocessing, Task 2 segmentation, real test data).

> Draft written from the code, docstrings and `reports/` – John to review and
> correct before it is final.

← Back to the [project README](../README.md) for installation and the whole pipeline.

**Run every command in this guide from the repository root**, not from inside `src/`.

Contents

1. [What this part of the system does](#1-what-this-part-of-the-system-does)
2. [Using it from code](#2-using-it-from-code)
3. [Files in `src/`](#3-files-in-src)
4. [Comparison scripts in `experiments/`](#4-comparison-scripts-in-experiments)
5. [The real handwritten test set](#5-the-real-handwritten-test-set)
6. [Results and where they are written up](#6-results-and-where-they-are-written-up)
7. [Tests](#7-tests)

---

## 1. What this part of the system does

```
photo of "896" ──► segmentation ──► 3 crops of the original photo ──► preprocessing ──► 3 images, 28×28, MNIST format
                    (Task 2)                                             (Task 1)
```

- **Segmentation** (`segmentation.py`) finds one box per digit, or per symbol
  for expressions, and returns them left to right. Selected method: `contours`.
- **Preprocessing** (`preprocessing.py`) turns one crop into the 28×28,
  white-on-black, 0–1 image the models were trained on. Selected
  configuration: `otsu_mnist_box`.

Both choices were made by comparing several techniques on generated **and**
real handwriting (section 6).

---

## 2. Using it from code

```python
import sys, cv2
sys.path.insert(0, "src")
from segmentation import segment_digits, segment_expression
from preprocessing import preprocess

image = cv2.imread("data/custom_samples/numbers/896_t1.jpg")

crops, boxes = segment_digits(image)            # boxes: (x, y, w, h), left to right
crops, boxes = segment_expression(image)        # also keeps + − × ÷ ( ) as separate boxes

x, y, w, h = boxes[0]
patch = preprocess(image[y:y + h, x:x + w], add_channel=True)   # (28, 28, 1), float32 in [0, 1]
```

Always cut the box from the **original photo** and give that to `preprocess`;
it expects dark ink on light paper. (`models/cnn/inference.py` wraps all of
this together with a model.)

**Main functions**

| Function | File | Returns |
|---|---|---|
| `segment_digits(image, method="contours", keep_operators=False, ...)` | `segmentation.py` | `(crops, boxes)`; methods: `contours`, `connected_components`, `projection`, `watershed` |
| `segment_expression(image, method="contours")` | `segmentation.py` | the same, with operators kept and the bar and dots of `÷` merged into one box |
| `annotate(image, boxes, labels=None)` | `segmentation.py` | a copy of the image with the boxes drawn |
| `preprocess(image, config="otsu_mnist_box", flatten=False, add_channel=False)` | `preprocessing.py` | one 28×28 image (or a 784-vector with `flatten=True`) |
| `preprocess_batch(images, config=...)` | `preprocessing.py` | a stacked array |
| `CONFIGS`, `SELECTED_CONFIG`, `METHODS`, `SELECTED_METHOD` | both | every technique that was compared, and the one selected |
| `load_digits()`, `load_numbers()` | `test_set.py` | `(filename, image, label)` for every real test photo |

---

## 3. Files in `src/`

| File | What it does | Run it with |
|---|---|---|
| `preprocessing.py` | Task 1: grayscale, median denoise, Otsu or adaptive threshold, 20×20 MNIST box, centre of mass, normalise; 8 named configurations | imported |
| `segmentation.py` | Task 2: four classical segmentation methods, operator support, `÷` handling | imported |
| `segmentation_ml.py` | Task 2: six model-based methods: sliding window with MLP / CNN / SVM / Random Forest, cut points, confidence split | imported |
| `digit_classifier.py` | the four classifiers used by `segmentation_ml.py`, with an extra "not a digit" class | `python src/digit_classifier.py` (trains and caches them, ~5 s) |
| `generate_number.py` | builds multi-digit number images (or expressions) from single-digit images, with ground truth | `python src/generate_number.py --count 30` |
| `prepare_data.py` | writes MNIST digits as PNG files to `data/raw/<digit>/` (input for the generator) | `python src/prepare_data.py` |
| `split_sheet.py` | cuts a photographed sheet with many handwritten numbers into one labelled image each | see the instructions at the top of the file |
| `test_set.py` | loads the real test photos and writes a manifest of what is in the set | `python src/test_set.py` |
| `download_symbols.py` | downloads the operator symbol datasets for the extension model (Thien) | `python src/download_symbols.py --check` |
| `verify_setup.py` | checks every required package is installed | `python src/verify_setup.py` |
| `show_progress.py` | DONE / TODO list of deliverables | `python src/show_progress.py` |

**`generate_number.py` options:** `--count N` (default 20), `--digits N`
(fixed length; default 2–5), `--difficulty {wide, tight, touching}` (gap
between digits), `--expressions` (arithmetic expressions instead of numbers),
`--seed`. Output: `data/generated/` (or `data/generated_expressions/`) with
`labels.csv`. Needs `python src/prepare_data.py` first.

---

## 4. Comparison scripts in `experiments/`

All outputs go to `reports/`.

| Command | Compares | Needs first | Output |
|---|---|---|---|
| `python experiments/show_techniques.py` | summary of every Task 1 and Task 2 technique, plus visual examples on real photos (no TensorFlow, 5 s) | – | terminal, `preprocessing_real_visual.png`, `segmentation_real_visual.png` |
| `python experiments/demo_one.py IMAGE` | the whole pipeline on one image, every step printed | – | terminal, `demo_one.png` |
| `python experiments/demo_one.py IMAGE --compare` | `contours` against the model-based segmenters on one image | `digit_classifier.py` | `demo_compare.png` |
| `python experiments/compare_preprocessing.py` | 8 preprocessing configurations, by retraining a CNN on each (slow) | – | `preprocessing_comparison.csv`, `preprocessing_visual.png` |
| `python experiments/demo_segmentation.py` | segmentation methods on generated numbers | `prepare_data.py`, `generate_number.py`, `digit_classifier.py` | `segmentation_comparison.csv`, `segmentation_demo/` |
| `python experiments/plot_model_results.py` | the four model-based segmenters in one chart | `demo_segmentation.py` | `segmentation_models_chart.png` |
| `python experiments/make_segmentation_figures.py` | the two segmentation figures used in the report | `generate_number.py`, `digit_classifier.py` | `segmentation_methods.png`, `segmentation_touching.png` |
| `python experiments/evaluate_custom.py` | preprocessing and segmentation on the **real** photos (3–4 min); `--all-methods`, `--all-configs`, `--model mlp` | – | `custom_preprocessing.csv`, `custom_segmentation.csv`, `custom_results_by_image.csv`, `custom_demo/` |
| `python experiments/evaluate_operators.py` | segmentation of expressions with and without operator support, and what it costs on digit photos | `generate_number.py --expressions --count 40` | `operator_segmentation.csv`, `operator_cost.csv`, `operator_segmentation.png` |

---

## 5. The real handwritten test set

`data/custom_samples/` (committed): **149 single digits** (14–15 per class) and
**99 numbers** of 2–5 digits, from six sheets by four writers
(`data/custom_samples/sheets.csv`, `sources.csv`). Numbers whose digits touch
are named `label_t1.jpg`, `label_t2.jpg`, …

The label is the part of the file name before the underscore:

```
data/custom_samples/digits/3_01.jpg         the digit 3
data/custom_samples/numbers/2026_01.jpg     the number 2026
```

To add more: write the numbers in rows on one sheet, photograph it, and cut it
with `src/split_sheet.py` (instructions at the top of the file), then run
`python src/test_set.py` to refresh `reports/custom_test_set.csv`.

The handwritten **expressions** used by the extension are in
`data/custom_samples/expressions/` and are described in
[`models/cnn/README.md`](../models/cnn/README.md#63-fine-tuning-on-the-teams-handwriting-ext_v3--ext_best).

---

## 6. Results and where they are written up

**Task 1 – preprocessing** (full table and reasoning in the comments above
`SELECTED_CONFIG` in `preprocessing.py`, and in `reports/report_task1_preprocessing.md`)

| Configuration | MNIST (LeNet, 5 seeds) | 149 real digits | 99 real numbers, end to end |
|---|---|---|---|
| **otsu_mnist_box (selected)** | 96.0 % | **81.9 %** | **62 %** |
| otsu_denoised_centered | 95.6 % | 79.9 % | 7 % |
| otsu | 96.7 % | 76.5 % | 1 % |
| otsu_denoised | 95.9 % | 76.5 % | 1 % |
| adaptive | 96.7 % | 63.1 % | 2 % |
| grayscale_only | 96.7 % | 61.1 % | 1 % |
| grayscale_mnist_box | 96.2 % | 38.3 % | 11 % |
| adaptive_mnist_box | 96.5 % | 33.6 % | 34 % |

**Task 2 – segmentation** (comments above `SELECTED_METHOD` in
`segmentation.py`, and `reports/report_task2_segmentation.md`)

| Method | 30 generated numbers | 99 real numbers | Real, read exactly |
|---|---|---|---|
| **contours (selected)** | 70 % | **75 %** | **62 %** |
| connected_components | 70 % | 75 % | 62 % |
| watershed | 50 % | 52 % | 34 % |
| projection | 83 % | 17 % | 5 % |

The six model-based segmenters were run on the same 99 photos:
`confidence_split` ties `contours` at 62 % read exactly; the others read 3–22 %.

**Measured limitations**

- Touching digits: 96 % of numbers with separate digits are split correctly,
  but only 12 % of numbers whose digits touch.
- Ruled paper: single digits on the lined sheet are read 30 % of the time,
  against 90 % on plain paper, because the line is kept as ink.
- Operators for expressions are kept only by `segment_expression()`; turning
  that on for plain numbers costs 4 points of digit-count accuracy
  (`reports/operator_cost.csv`).

---

## 7. Tests

```bash
python -m pytest tests -q
```

About 120 tests for preprocessing, both segmentation modules, the sheet splitter
and the symbol downloader. About 24 are skipped until `prepare_data.py`,
`generate_number.py` and `digit_classifier.py` have been run once.
