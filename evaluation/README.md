# `evaluation/` – baselines, model comparison and system evaluation

Lane: **Baselines & Evaluation** (Task 3 classical baselines, Task 4 evaluation
and testing, model selection, critical analysis). Allocated to Liam; written by
Thien when that work was not delivered.

← Back to the [project README](../README.md) for installation and the whole pipeline.

**Run every command in this guide from the repository root.**

Contents

1. [What this lane answers](#1-what-this-lane-answers)
2. [Quick start](#2-quick-start)
3. [Files](#3-files)
4. [The classical baselines](#4-the-classical-baselines)
5. [The test sets](#5-the-test-sets)
6. [Metrics](#6-metrics)
7. [Failure analysis](#7-failure-analysis)
8. [Outputs](#8-outputs)
9. [Tests](#9-tests)
10. [Report sections](#10-report-sections)

---

## 1. What this lane answers

1. **Which recognition technique should the system use?** Six models from three
   families – classical machine learning (HOG + SVM, HOG + Random Forest), a
   simple neural network (MLP) and three CNNs (LeNet-5, ResNet, VGG-deep) – are
   scored on the same inputs, and the differences are tested for significance.
2. **How well does the final system work?** Per-digit and whole-number accuracy,
   kept separate as the specification asks, on generated numbers and on real
   handwriting; plus the extension's symbol and calculated-result accuracy.
3. **Where and why does it fail?** Every failure is put in one category
   (segmentation or recognition) with the evidence in tables and figures.

---

## 2. Quick start

```bash
python evaluation/baselines.py        # train HOG + SVM and HOG + Random Forest (~30–40 min, once)
python evaluation/generated_set.py    # build 600 test numbers from MNIST test digits (~10 s, once)
python evaluation/evaluate_all.py     # every model on every test set (5–10 min)
python evaluation/error_analysis.py   # failure categories, gallery, confidence (~1 min)
```

Results land in `evaluation/results/`. Start with `model_comparison.md` and
`failure_analysis.md`.

The neural models are the frozen ones in `models/checkpoints/` (trained in
`models/cnn/`, see [its guide](../models/cnn/README.md)); nothing here retrains them.

---

## 3. Files

| File | What it does |
|---|---|
| `common.py` | The six compared models behind one interface (`predict_proba` on 28×28 images), the shared MNIST split, Wilson confidence intervals |
| `baselines.py` | HOG features; tunes and trains HOG + SVM and HOG + Random Forest on validation; saves them to `models/checkpoints/hog_*.joblib` |
| `generated_set.py` | Builds the Task 4 generated-number test set from MNIST **test** digits |
| `evaluate_all.py` | The harness: every model on MNIST, real digits, real numbers and generated numbers; McNemar tests; tables and charts |
| `error_analysis.py` | Puts every failure of the final pipeline in a category; writer, sheet and spacing breakdowns; confidence analysis; extension breakdown |
| `test_evaluation.py` | Unit tests for all of the above |
| `report_comparison_of_techniques.md` | Report section: comparison of ML techniques and model selection |
| `report_system_performance.md` | Report section: overall system functions and performance, with the critical analysis |
| `results/` | Everything the scripts write (section 8) |

---

## 4. The classical baselines

**Features: HOG** (histogram of oriented gradients, `skimage.feature.hog`).
Each image is divided into cells and every cell is described by the directions
of its edges – close to how a digit is recognised by its strokes. Two cell sizes
were compared on validation: 7×7-pixel cells (324 values per image) and
4×4-pixel cells (1,296 values). The finer one won for both classifiers.

**Classifiers:**

- **SVM** with an RBF kernel. Grid: C ∈ {1, 5, 10, 20} × γ ∈ {scale/2, scale, scale×2}.
- **Random Forest.** Grid: 200 trees × max_features ∈ {√n, 0.2} × min_samples_leaf ∈ {1, 2}.

**Selection** uses the same fixed MNIST split as every neural model
(`models/cnn/mnist_loader.py`: 54,000 train / 6,000 validation / 10,000 test).
Settings are chosen on validation accuracy (grid search on 12,000 training
images), then the chosen setting is trained on all 54,000 and the test set is
scored once. Every setting tried and its validation accuracy:
`results/baseline_tuning.csv`; the chosen settings: `models/checkpoints/hog_*.source.json`.

```bash
python evaluation/baselines.py               # both
python evaluation/baselines.py --model svm   # one
python evaluation/baselines.py --quick       # 2-minute check of the code, nothing logged
```

Both baselines are also options in the GUI (*SVM*, *Random Forest*), pretrained
or trained there on 10,000 images with these settings
([GUI backend](../models/cnn/README.md#8-gui-backend)).

---

## 5. The test sets

| Set | Size | Source | Notes |
|---|---|---|---|
| MNIST test | 10,000 digits | standard MNIST test split | never used for training or selection by any model |
| MNIST shifted | 10,000 digits | the same, each slightly rotated / shifted / zoomed (fixed seed) | closer to what segmentation produces from photos |
| Real digits | 149 photos | `data/custom_samples/digits/` | 4 writers, 6 sheets |
| Real numbers | 99 photos, 2–5 digits | `data/custom_samples/numbers/` | 25 with touching digits (`*_t*.jpg`) |
| Generated numbers | 600, 2–5 digits | `evaluation/generated_set.py` | MNIST **test** digits; 200 each at wide, tight and touching spacing |
| Expressions | 408 photos | `data/custom_samples/expressions/ready/`, sheets 3–4 | 14 writers never used in training; scored by `models/cnn/evaluate_expressions.py` |

**Why a separate generated set.** `src/generate_number.py` builds numbers from
`data/raw/`, which holds MNIST *training* digits – the images every model
learned from. Scoring on those would overstate every model. `generated_set.py`
uses the same layout code and spacings with MNIST *test* digits instead.

---

## 6. Metrics

| Metric | Definition |
|---|---|
| Accuracy (single digits) | share of digits classified correctly |
| Precision / recall / F1 per digit | from the confusion matrix on MNIST test; macro-F1 is their mean |
| **Per-digit accuracy** (numbers) | 1 − edit distance ÷ number of true digits: a wrong, missing or extra digit is one error |
| Per-digit accuracy when the count is right | digits right ÷ digits, only on numbers segmented into the right number of boxes – isolates recognition |
| **Whole-number accuracy** (numbers) | share of numbers with every digit right |
| Digit count right | share of numbers segmented into exactly the right number of boxes – a property of segmentation, identical for every model |
| 95 % confidence interval | Wilson score interval; shown wherever a set is small (the real photos) |
| McNemar's exact test | paired comparison of two models on the same images: only the images one gets right and the other wrong count. p < 0.05 = a real difference |

Segmentation and preprocessing are the system's own and run once per image:
the repaired segmentation (`inference.segment_number()`: ruled lines removed,
`contours`, touching digits cut where `cnn_best` reads the pieces best) and
`otsu_mnist_box`. Every model reads the same crops, so differences between
models are differences in recognition only.

---

## 7. Failure analysis

`error_analysis.py` runs the final pipeline (repaired segmentation →
otsu_mnist_box → VGG-deep) and puts every failed number in exactly one category:

| Category | Rule | Stage at fault |
|---|---|---|
| merged | fewer boxes than digits, no digit-sized ink discarded | segmentation – touching digits cut as one |
| dropped | fewer boxes than digits, and the ink thrown away adds up to a digit (pieces grouped, ≥ half a digit tall) | segmentation – a faint digit broken into fragments, each too small for the noise filter |
| split | more boxes than digits | segmentation |
| misread | right number of boxes, a digit wrong | recognition |

Ruled lines and underlines (ink blobs more than 8× wider than tall) are ignored
when looking for dropped digits. The rules were checked against the data: no
"merged" case occurs at the wide or tight spacing, where digits cannot touch.

It also breaks failures down by writer, sheet and spacing, lists the most
frequent misreadings, measures whether low model confidence could flag wrong
digits, and splits the extension's failures into segmentation and recognition.

---

## 8. Outputs

All in `evaluation/results/` (committed – they are the evidence for the report):

| File | Contents |
|---|---|
| `model_comparison.md` / `.csv` | the main comparison tables |
| `comparison_chart.png` | every model on every test set |
| `confusion_mnist.png` | confusion matrix of every model on MNIST test |
| `mnist_per_class.csv` | precision / recall / F1 per digit per model |
| `significance.csv` | McNemar tests against VGG-deep |
| `numbers_by_image.csv` | every real and generated number as read by every model |
| `baseline_tuning.csv` | every baseline setting tried, with validation accuracy |
| `failure_analysis.md` | failure categories and breakdowns |
| `failures.csv` | every failed number with its category |
| `failure_gallery.png` | examples of each failure category on real photos |
| `confidence.png` | model confidence on right vs wrong digits |

Not committed: `data/generated_test/` (rebuilt identically from its seed).

---

## 9. Tests

```bash
python -m pytest evaluation -q
```

Metrics (edit distance, per-digit vs whole-number, McNemar, Wilson), HOG
features, saving and reloading a baseline, and the failure categories on
synthetic images. No trained model or download needed.

---

## 10. Report sections

- [`report_comparison_of_techniques.md`](report_comparison_of_techniques.md) –
  "Comparison of ML techniques": the three families, how they were trained and
  selected, results, significance, and the justified choice of model.
- [`report_system_performance.md`](report_system_performance.md) – "Overall
  system functions and performance": what the system does, per-digit and
  whole-number accuracy on generated and real input, the extension, and the
  critical analysis of failures.
