# `models/cnn/` – recognition models, training and evaluation

Owner: **Thien** (Task 3 deep models, extension model). This folder turns a
28×28 image into a class (`0–9`, or `0–9 + − × ÷ ( )` for the extension),
and holds everything used to train, tune, compare and evaluate those models.

← Back to the [project README](../../README.md) for installation and the
whole pipeline.

**Run every command in this guide from inside this folder:**

```bash
cd models/cnn
```

Contents

1. [The frozen models (what the system uses)](#1-the-frozen-models-what-the-system-uses)
2. [Using a model from code: `inference.py`](#2-using-a-model-from-code-inferencepy)
3. [Files in this folder](#3-files-in-this-folder)
4. [Training and tuning the digit models](#4-training-and-tuning-the-digit-models)
5. [Comparing the four architectures](#5-comparing-the-four-architectures)
6. [The extension model (16 classes)](#6-the-extension-model-16-classes)
7. [Evaluation scripts](#7-evaluation-scripts)
8. [GUI backend](#8-gui-backend)
9. [Where every output goes](#9-where-every-output-goes)
10. [Tests](#10-tests)
11. [Design decisions worth knowing](#11-design-decisions-worth-knowing)

---

## 1. The frozen models (what the system uses)

All in `models/checkpoints/` and committed to Git. Each `.keras` file has a
`.labels.json` (class names in output order) and a `.source.json` (the training
run it was copied from) beside it.

| Name | Classes | Architecture | Parameters | Used by | Source run |
|---|---|---|---|---|---|
| **`cnn_best`** | 0–9 | VGG-deep | 0.59 M | GUI option *CNN*, number reading | `s3_B5_bs_64` (Sprint 3 tuning) |
| **`ext_best`** | 0–9 + − × ÷ ( ) | VGG-deep | 0.59 M | GUI option *CNN + operators*, expressions | `ext_v3` (fine-tuned on team handwriting) |
| `ext_v2` | 16 | VGG-deep | 0.59 M | before/after comparison only | `ext_v2` (MNIST + symbols, before fine-tuning) |
| `lenet_best` | 0–9 | LeNet-5 | 0.06 M | model comparison | `pm_lenet_best` |
| `resnet_best` | 0–9 | small ResNet | 0.17 M | model comparison | `pm_resnet_best` |
| `mlp_best` | 0–9 | MLP (no convolutions) | 1.34 M | model comparison | `pm_mlp_best` |
| `cnn_lenet`, `cnn_vgg_deep`, `cnn_resnet`, `mlp_simple`, `cnn_shallow`, `cnn_vgg_small` | 0–9 | – | – | Sprint 2 same-settings runs | the Sprint 2 scripts (§3) |

In the class files, `*` means multiplication (written × or *) and `/` means
division (written ÷).

---

## 2. Using a model from code: `inference.py`

`Recogniser` loads a frozen model with its class names and runs the project's
own segmentation and preprocessing, so callers never repeat that logic.

```python
import cv2
from inference import Recogniser                 # sys.path.insert(0, "models/cnn") from the repo root

image = cv2.imread("../../data/custom_samples/numbers/896_t1.jpg")

digits = Recogniser("cnn_best")
text, details = digits.read_image(image)          # '896'

expr = Recogniser("ext_best")
text, details = expr.read_image(image)            # reads + − × ÷ ( ) as well
text, details = expr.read_image(image, allowed=Recogniser.DIGITS)   # digits only
```

- `read_image(image, allowed=None, method=None)` segments the photo, cuts each
  box from the original photo, preprocesses it to 28×28 and classifies it.
  Numbers use `segment_number()`: John's repaired segmentation by default
  (`src/segmentation_split.py` – touching digits cut where `cnn_best` reads
  the pieces best; `cnn_best` chooses the cuts for every model, so all models
  get the same crops), or `method="contours"` for the Task 2 selection.
  Models with operator classes use `segment_expression`.
  It returns the text and one dict per symbol: `label`, `confidence`, `top3`,
  `box` (x, y, w, h) and `input` (the 28×28 image the model saw).
- `classify(patches)` / `probabilities(patches, allowed)` work on ready-made
  28×28 arrays.
- `allowed` restricts the answer to some classes (for example digits only for
  a plain number) and renormalises the probabilities.
- The model name can be a frozen name (`"cnn_best"`), a tuning run
  (`"tuning/ext_v3"`) or a path to a `.keras` file.

From the command line:

```bash
python inference.py ../../data/custom_samples/numbers/896_t1.jpg                       # cnn_best
python inference.py ../../data/custom_samples/expressions/ready/e090_s3w1.jpg --model ext_best
python inference.py IMAGE --model ext_best --digits-only
```

---

## 3. Files in this folder

**Models and data**

| File | What it does |
|---|---|
| `model_zoo.py` | Builds the four architectures by name: `mlp`, `lenet`, `vgg_deep`, `resnet` (`build_model(arch, num_classes, dropout)`). |
| `mnist_loader.py` | MNIST with a fixed 54k / 6k / 10k train / validation / test split, scaled to 0–1. Used by every training script. |
| `datasets.py` | Registry of training datasets: `mnist` (10 classes) and `mnist+symbols` (16 classes). Defines the class order. |
| `symbols_data.py` | Builds `mnist+symbols` from the downloaded operator images in `data/symbols/`: same preprocessing as photos, stroke width matched to MNIST, classes balanced, cached. |
| `real_symbols.py` | Cuts labelled symbols out of the team's expression photos for fine-tuning, split by writer (train / validation / test). |
| `augmentation.py` | Rotation / shift / zoom presets (`none`, `light`, `medium`, `strong`), the shifted test set, and the style augmentation (stroke width, hard / sharpened / blurred edges, contrast) plus synthetic crossed 7s and flagged 1s. |

**Training**

| File | What it does |
|---|---|
| `trainer.py` | The config-driven trainer used since Sprint 3: one run from the command line or a whole grid from a JSON file. Logs every run. |
| `configs/sprint3_tuning.json` | Sprint 3 grid for VGG-deep: stage A augmentation, stage B learning rate / dropout / batch size, optional stage C. |
| `configs/sprint3_per_model.json` | Each of LeNet, ResNet and MLP at its own best settings (stage P). |
| `summarise_tuning.py` | Table and chart of all runs on one dataset; `--freeze` copies the best one to `checkpoints/`. |
| `ext_finetune.py` | Fine-tunes the 16-class model on the team's handwriting (§6.3). |
| `experiment_logger.py` | Appends one row per run to the shared CSV log. |
| `cnn_lenet.py`, `cnn_vgg_deep.py`, `cnn_resnet.py`, `mlp_simple.py` | The original Sprint 2 scripts, one per architecture, with shared fixed settings. Kept for the same-settings comparison. |

**Evaluation and use**

| File | What it does |
|---|---|
| `inference.py` | `Recogniser`: load a frozen model and read an image (§2). |
| `evaluate_classes.py` | Precision / recall / F1 per class and confusion matrices on the MNIST(+symbols) test set, clean and shifted. |
| `evaluate_real.py` | Every frozen model on the team's real digit and number photos. |
| `evaluate_expressions.py` | The extension model on the team's handwritten expressions: exact match, symbol accuracy, calculated result. |
| `compare_models.py` | Accuracy and loss curves of the four architectures in one chart. |
| `gui_backend.py` | The functions `gui.py` calls for every model option (§8). |

**Tests:** every `test_*.py` file, plus `conftest.py` (puts this folder on the
import path) and `cnn_test.py` (Sprint 2 tests).

---

## 4. Training and tuning the digit models

MNIST downloads automatically the first time (11 MB).

```bash
# One run: any architecture, augmentation, learning rate, batch size, dropout, epochs
python trainer.py --arch vgg_deep --aug light --lr 1e-3 --batch-size 64 --run-id my_test

# A 2-epoch smoke test on a small subset (nothing is logged)
python trainer.py --arch lenet --quick --run-id smoke

# The Sprint 3 grid, one stage at a time (each VGG run takes 10–20 min on a CPU)
python trainer.py --grid configs/sprint3_tuning.json --stage A
python trainer.py --grid configs/sprint3_tuning.json --stage B
python trainer.py --grid configs/sprint3_tuning.json --stage B --skip-existing   # resume

# Table + chart of all digit runs; --freeze copies the best to checkpoints/cnn_best.keras
python summarise_tuning.py
python summarise_tuning.py --freeze
```

`trainer.py` options: `--arch {mlp, lenet, vgg_deep, resnet}`,
`--dataset {mnist, mnist+symbols}`, `--aug {none, light, medium, strong}`,
`--lr`, `--batch-size`, `--dropout`, `--epochs` (maximum; early stopping with
patience 5 on validation accuracy), `--seed`, `--run-id`, and for grids
`--grid`, `--stage`, `--only RUN_ID…`, `--skip-existing`, `--quick`.

Every run writes `models/checkpoints/tuning/<run>.keras` (not committed),
`models/experiments/tuning/<run>_summary.json`, `_history.json`,
`_history.png`, and a row in `models/experiments/tuning_log.csv`.

**How the digit model was chosen.** Stage A (augmentation, everything else
fixed) picked `light`; stage B varied one hyperparameter at a time from that
baseline and picked batch size 64. Selection is always by **validation**
accuracy; the test set is reported but never used to choose.
Results: `models/experiments/tuning_table.md` and `tuning_comparison.png`.

Every run reports two test numbers: the normal MNIST test set, and a
*shifted* copy (each image slightly rotated, shifted and zoomed, the same for
every run) that is closer to what segmentation produces from real photos.

---

## 5. Comparing the four architectures

Two comparisons, kept separate on purpose:

```bash
# Same settings for every model (the Sprint 2 runs)
python compare_models.py
# Each model at its own best settings
python trainer.py --grid configs/sprint3_per_model.json     # only if you want to retrain them
python compare_models.py --set best
```

Outputs: `models/experiments/compare_accuracy.png` / `compare_loss.png`, and
`compare_best_accuracy.png` / `compare_best_loss.png` / `compare_best_table.md`.
Never put a tuned model next to an untuned one in the same table.

| Model (best settings) | MNIST test | Shifted test | Real single digits | Real numbers exact |
|---|---|---|---|---|
| MLP | 98.99 % | 96.89 % | 57.0 % | 29.3 % |
| LeNet-5 | 99.29 % | 98.09 % | 83.9 % | 64.6 % |
| ResNet | 99.57 % | 99.54 % | 83.9 % | 58.6 % |
| **VGG-deep (`cnn_best`)** | **99.62 %** | 99.47 % | **86.6 %** | **67.7 %** |

---

## 6. The extension model (16 classes)

### 6.1 Operator symbol data

From the **repository root** (not this folder):

```bash
python src/download_symbols.py --check     # what is present
python src/download_symbols.py --clean     # download HASYv2, read the Kaggle set, build data/symbols/
```

HASYv2 downloads automatically. The Kaggle *Handwritten math symbols* set
needs a manual download (login required): extract `data.rar` into
`data/downloads/xainano/`. `symbols_data.py` then preprocesses the images like
photos, thickens strokes to MNIST width, caps each class at 6,000 images and
builds 80 / 10 / 10 splits per class.

### 6.2 Base model: MNIST + symbols (`ext_v2`)

```bash
python trainer.py --dataset mnist+symbols --arch vgg_deep --aug light --batch-size 64 --run-id ext_v2
python summarise_tuning.py --dataset mnist+symbols
```

Result: 99.57 % on the MNIST + symbols test set. On real handwriting it read
brackets as `1` (the symbol images have soft edges and the photos have hard
edges) and could not read the team's crossed 7 and flagged 1.

### 6.3 Fine-tuning on the team's handwriting (`ext_v3` = `ext_best`)

```bash
python real_symbols.py                                   # build the real-symbol cache (~1 min)
python ext_finetune.py --run-id ext_v3                   # full recipe (~30 min on a CPU)
python ext_finetune.py --run-id ext_v3_noreal --real-repeat 0     # ablation: no real data
python ext_finetune.py --run-id ext_v3_nostyle --no-style         # ablation: no style augmentation
python summarise_tuning.py --dataset mnist+symbols+real  # table of the fine-tuned runs
```

**Data.** `real_symbols.py` segments each of the 823 expression photos in
`data/custom_samples/expressions/ready/` with the GUI pipeline. If the number
of boxes equals the length of the true expression, box *i* is labelled with
character *i*; otherwise the photo is skipped (73 were). This gives 5,757
real symbols, split **by writer** so the test writers are never seen:

| Split | Writers | Symbols | Used for |
|---|---|---|---|
| train | sheets 1–2, writers 1–6 | 2,500 | training (repeated ×6 per epoch) |
| validation | sheets 1–2, writer 7 | 441 | choosing the epoch and settings |
| test | sheets 3–4, all 14 writers | 2,816 | final numbers only, used once |

**Training set per epoch (98,453 images):** MNIST + symbols (80,453); 1,500
synthetic crossed 7s and 1,500 synthetic flagged 1s drawn onto MNIST digits;
the 2,500 real training symbols repeated 6 times.

**Augmentation**, random for every image in every batch (`augmentation.style_augment`):

| Step | Options (probability) |
|---|---|
| Geometry (`light`) | rotate ±8°, shift ±8 %, zoom ±8 % |
| Stroke width | unchanged (60 %), thicker by dilation (25 %), thinner by erosion (15 %) |
| Edge style | unchanged (25 %), hard binary `sigmoid((x − t)·30)` with random t in 0.25–0.6 (30 %), sharpened by unsharp mask (15 %), Gaussian blur σ = 0.5 / 0.8 / 1.1 (10 % each) |
| Contrast | renormalised to peak 1 (80 %), peak 0.7–1.0 (20 %) |

**Settings.** Start from `ext_v2` weights; Adam, learning rate 3·10⁻⁴ halved
whenever real-validation accuracy stops improving (minimum 2·10⁻⁵); batch 64;
8 epochs; keep the epoch with the best real-validation accuracy (epoch 7).

**Results on the 14 unseen writers (408 expressions):**

| Model | Exact expression | Symbol accuracy |
|---|---|---|
| `ext_v2` (before) | 70.8 % | 94.2 % |
| style augmentation, no real data | 80.9 % | 96.3 % |
| real data, no style augmentation | 86.5 % | 97.3 % |
| **`ext_v3` = `ext_best`** | **87.5 %** | **97.4 %** |

Full write-up for the report: `models/experiments/eval/retraining_report.md`.

**Freezing.** `ext_best` was copied from `ext_v3` by hand (see
`checkpoints/ext_best.source.json`). `summarise_tuning.py --freeze` refuses to
replace a frozen model whose source run is not in the table it is looking at,
so `--dataset mnist+symbols --freeze` cannot silently put `ext_v2` back. Use
`--dataset mnist+symbols+real --freeze` after new fine-tuning runs, or add
`--force` if you really mean to replace it.

---

## 7. Evaluation scripts

| Command | Data | Measures | Output (`models/experiments/eval/`) |
|---|---|---|---|
| `python evaluate_classes.py` | MNIST(+symbols) test set, clean and shifted | precision / recall / F1 per class, confusion matrices | `<model>_report.md`, `_per_class.csv`, `_confusion.png`, `_confusion_shifted.png` |
| `python evaluate_classes.py --model ext_v2` | same | the extension model before fine-tuning | `ext_v2_*` |
| `python evaluate_real.py` | 149 real digit photos, 99 real number photos | single-digit accuracy (95 % CI), exact numbers, digit count | `real_handwriting.md`, `.csv`, `_errors.csv`, `real_vs_mnist.png`, `real_errors_cnn_best.png` |
| `python evaluate_real.py --runs RUN…` | same | also any tuning run kept locally in `checkpoints/tuning/` | `real_handwriting_runs.csv` |
| `python evaluate_expressions.py` | 408 expression photos, sheets 3–4 | exact expression, symbol accuracy, result, per symbol, per writer | `expressions_report.md`, `expressions_results.csv` |
| `python evaluate_expressions.py --model ext_v2 --tag _ext_v2_test` | same | the model before fine-tuning | `expressions_report_ext_v2_test.md` |

`evaluate_classes.py` needs `data/symbols/` (§6.1) for the 16-class models;
`--model cnn_best` works with MNIST alone.

`evaluate_expressions.py` options: `--model`, `--sheets` (default `3,4`, the
writers never used in training; `all` includes the training writers and must
not be used for reported numbers), `--tag` (suffix for output files),
`--limit N`, `--soften` / `--no-soften` (force the input blur on or off).

How the expression metrics are computed:

- **Exact expression** – the whole string read equals the truth.
- **Symbol accuracy** – 1 − edit distance / true length (counts wrong, missing and extra symbols).
- **Result correct** – the string read is calculated by a small evaluator in
  `evaluate_expressions.py` (precedence and brackets, no `eval()`) and compared
  with the true answer.
- **Per-symbol accuracy** – only on expressions where segmentation found the
  right number of symbols, so each symbol can be paired with its truth.

---

## 8. GUI backend

`gui.py` (repository root) imports `gui_backend.py` for all five model options.

| GUI option | Pretrained | Train new |
|---|---|---|
| *CNN* | `cnn_best.keras` | chosen CNN architecture, GUI settings, MNIST |
| *CNN + operators* | `ext_best.keras` | chosen CNN architecture, GUI settings, MNIST + symbols |
| *MLP* | `mlp_best.keras` | MLP architecture, GUI settings, MNIST |
| *SVM* | `hog_svm.joblib` | HOG + SVM with the settings tuned in `evaluation/baselines.py`, 10,000 MNIST images (~3 min) |
| *Random Forest* | `hog_rf.joblib` | HOG + Random Forest, tuned settings, 10,000 MNIST images (< 1 min) |

| Function | Called when | Does |
|---|---|---|
| `load_model(model_name, path=None)` | *Pretrained* selected | loads the frozen model above (or a chosen `.keras` / `.joblib` file) as a `Recogniser`; refuses a file of the wrong kind |
| `train_model(model_name, params, on_epoch, stop_event)` | *Train* pressed | neural options: trains through `trainer.py`, reports every epoch, honours *Stop*; SVM / Random Forest: one fit, reported as one step. Saves to `models/gui_runs/` |
| `recognise(model, pil_image, segmentation=None)` | *Recognise Number* pressed | returns the text, per-symbol confidences, boxes, 28×28 thumbnails and the binary image for the GUI tabs; `segmentation` is the *Segment* choice (`SEGMENTATIONS`: `repaired`, the default, or `contours`) |
| `generate_from_folder(folder, text, n_digits, expression)` | *Generate* pressed | builds the input image from a folder of digit images (`src/number_from_folder.py`); returns the image and the number it shows |

The SVM and Random Forest are wrapped by `ClassicalModel`, which gives them the
one method `Recogniser` calls (`predict` on 28×28 patches). Every option
therefore reads a photo through the same segmentation and preprocessing, and
the GUI's readings of the real test photos match
`evaluation/results/numbers_by_image.csv` exactly.

---

## 9. Where every output goes

| Path | Contents | Committed |
|---|---|---|
| `models/checkpoints/*.keras` | frozen models (§1) | yes |
| `models/checkpoints/tuning/` | every training run's model | no (7 MB each) |
| `models/experiments/tuning_log.csv` | one row per trainer / fine-tune run | yes |
| `models/experiments/experiment_log.csv` | Sprint 2 runs | yes |
| `models/experiments/tuning/` | summary, history and curve of every run | yes |
| `models/experiments/tuning_table.md`, `extension_table.md`, `extension_finetune_table.md` | run tables per dataset | yes |
| `models/experiments/compare_*` | architecture comparison charts and table | yes |
| `models/experiments/eval/` | all evaluation reports (§7) and `retraining_report.md` | yes |
| `models/gui_runs/` | models trained from the GUI | no |
| `data/custom_samples/expressions/real_symbols_cache.npz` | cache built by `real_symbols.py` | no |

---

## 10. Tests

```bash
python -m pytest -q          # about 120 tests, 1–2 minutes; no trained model or download needed
```

They cover the trainer (including Stop from the GUI), the dataset builders,
augmentation, the writer split, `Recogniser` (with a fake model), the GUI
backend, the evaluation helpers and the tuning summary / freeze logic.

---

## 11. Design decisions worth knowing

- **Class names live next to the model.** Never assume output index = digit:
  read `<model>.labels.json`. For the 16-class models, indices 10–15 are
  `+ - * / ( )`.
- **Crop the original photo, not the binary image.** Preprocessing expects
  dark ink on light paper; `Recogniser` does this for you.
- **`input_blur` flag.** `ext_v2` only read brackets if each 28×28 input was
  blurred slightly (`inference.soften`). The fine-tuned `ext_best` does not
  need it; its `.labels.json` says `"input_blur": false` and `Recogniser`
  follows that flag automatically.
- **Selection never uses a test set.** Digit models are chosen on MNIST
  validation accuracy; the fine-tuned model on the validation writers. Real
  test photos and the held-out expression sheets are only reported.
- **Datasets are kept apart.** `summarise_tuning.py` compares runs only within
  one dataset (`mnist`, `mnist+symbols`, `mnist+symbols+real`), because a
  10-class and a 16-class test accuracy are not comparable.
