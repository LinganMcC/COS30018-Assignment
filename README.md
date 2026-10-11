# COS30018 – Handwritten Number Recognition System (HNRS)

Option B group project. The system reads a photo of a handwritten number
(for example `94026`) and outputs the number.

**Team**

- **John** – Data & Vision: preprocessing (Task 1), segmentation (Task 2), test images
- **Thien** – Deep Models: CNN models, augmentation, hyperparameter tuning (Task 3)
- **Liam** – Baselines & Evaluation: SVM / MLP baselines, evaluation (Task 3 + Task 4)
- **Russell** – Systems & Extension: GUI, pipeline wiring, arithmetic extension

---

## 1. How the system works

```
photo ─► preprocessing ─► segmentation ─► digit recognition ─► number
         (John, Task 1)   (John, Task 2)   (Thien + Liam,        "94026"
                                            Task 3)
```

1. **Preprocessing** cleans each digit image and turns it into the 28×28
   format the models were trained on (`src/preprocessing.py`).
2. **Segmentation** cuts a number image into one box per digit, left to
   right (`src/segmentation.py`).
3. **Recognition** classifies each 28×28 digit with a trained model
   (`models/checkpoints/*.keras`).
4. The digits are joined back into the number.

---

## 2. Status (as of 11 Oct, end of Sprint 3)

**Done**

- [x] Task 1 – 8 preprocessing configurations compared; selected: `otsu_mnist_box`
- [x] Task 2 – 4 classical + 4 model-based (sliding-window) segmentation methods compared; selected: `contours`
- [x] Real handwritten test photos in `data/custom_samples/`
- [x] 4 recognition models trained on MNIST: LeNet-5, VGG-deep, ResNet, MLP
- [x] Config-driven trainer with data augmentation (`models/cnn/trainer.py`)
- [x] Tuning stage A (augmentation) and stage B (learning rate / dropout / batch size)
- [x] Final digit CNN frozen: `models/checkpoints/cnn_best.keras` (deeper VGG, batch 64, 99.62 % test)
- [x] Extension dataset: operator symbols from HASYv2 + Kaggle (`src/download_symbols.py`)
- [x] First 16-class model (digits + `+ - * / ( )`): `models/checkpoints/ext_best.keras`

**Still open**

- [ ] Per-model tuning of LeNet / ResNet / MLP and the "best settings" comparison – Thien
- [ ] Per-symbol accuracy and confusion matrix for `ext_best.keras` – Thien + Liam
- [ ] GUI still uses placeholder functions; `predict_digits` returns random digits – Russell
- [ ] Segmentation of operators (see section 8) – John
- [ ] Team's own handwritten operator samples for real-world testing – Russell + team

Run `python src/show_progress.py` for a live DONE/TODO list.

---

## 3. Setup (do this once)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python src/verify_setup.py
```

`verify_setup.py` should report every required package as installed.
TensorFlow is only needed for training and for anything that uses the CNN.

---

## 4. First run after cloning

Datasets are not stored in git, so a fresh clone has to rebuild them.
Run these in order from the repo root – each one needs the one before it.

**Step 1 – Export MNIST digits as image files** (into `data/raw/`)
```bash
python src/prepare_data.py
```

**Step 2 – Generate multi-digit number images with labels** (into `data/generated/`)
```bash
python src/generate_number.py --count 30
```

**Step 3 – Train the small classifiers used by the model-based segmenters**
```bash
python src/digit_classifier.py
```

**Step 4 – Run all tests**
```bash
python -m pytest tests/ -q
cd models/cnn
python -m pytest -q
cd ../..
```

You can now run any script in sections 5–7.

---

## 5. Task 1 & 2 – preprocessing and segmentation (John)

All outputs go into `reports/`.

```bash
# Whole pipeline on ONE image, printing every step (good for debugging)
python experiments/demo_one.py data/custom_samples/numbers/896_t1.jpg

# Task 1 – compare preprocessing configurations (slow: retrains a CNN each time)
python experiments/compare_preprocessing.py

# Task 2 – compare segmentation methods on generated numbers
python experiments/demo_segmentation.py
python experiments/plot_model_results.py
python experiments/make_segmentation_figures.py

# Both comparisons on REAL handwriting (data/custom_samples/)
python experiments/evaluate_custom.py

# Every technique and result in one summary (fast, no TensorFlow)
python experiments/show_techniques.py
```

**Adding your own handwriting:** put the label before the underscore in the filename.

```
data/custom_samples/digits/3_01.jpg         one digit, label 3
data/custom_samples/numbers/94026_01.jpg    a whole number, label 94026
```

Several numbers written on one sheet can be cut up automatically with
`src/split_sheet.py` (see the instructions at the top of that file).

---

## 6. Task 3 – training the models (Thien)

All model code lives in `models/cnn/`. Run these commands **from inside that folder**.

**The four models**

- `lenet` – LeNet-5, small and fast (`cnn_lenet.py`)
- `vgg_deep` – 3-block VGG, the main candidate (`cnn_vgg_deep.py`)
- `resnet` – small ResNet, most accurate but slowest (`cnn_resnet.py`)
- `mlp` – fully connected baseline, no convolutions (`mlp_simple.py`)

The `cnn_*.py` / `mlp_simple.py` scripts are the original Sprint 2 versions.
For new training, use the shared trainer instead:

```bash
cd models/cnn

# One training run
python trainer.py --arch vgg_deep --aug light --run-id my_test

# A tuning stage from the grid (configs/sprint3_tuning.json)
python trainer.py --grid configs/sprint3_tuning.json --stage B

# Quick 2-epoch check that everything works (nothing is logged)
python trainer.py --grid configs/sprint3_tuning.json --stage B --quick

# Results table + chart (add --freeze to copy the best run to checkpoints/cnn_best.keras)
python summarise_tuning.py
```

`--freeze` picks the best run by validation accuracy across **all** digit runs, so
only use it when you mean to replace the frozen model.

**Comparing the four models** – two comparisons, both worth showing in the report:

```bash
# Controlled: every model with the same shared settings (the Sprint 2 runs)
python compare_models.py

# Each model at its best: tuned VGG + the per-model runs below
python trainer.py --grid configs/sprint3_per_model.json
python compare_models.py --set best
```

The first gives `compare_accuracy.png` / `compare_loss.png`; the second gives
`compare_best_accuracy.png` / `compare_best_loss.png` and `compare_best_table.md`.
Never mix the two (e.g. tuned VGG next to untuned LeNet) in one table.

**Extension: digits + operator symbols (16 classes)**

```bash
# From the repo root, once: download the symbols into data/symbols/ (not committed)
python src/download_symbols.py --check
python src/download_symbols.py --clean

# From models/cnn/: train, then summarise extension runs separately from digit runs
python trainer.py --dataset mnist+symbols --arch vgg_deep --aug light --batch-size 64 --run-id ext_v2
python summarise_tuning.py --dataset mnist+symbols --freeze     # -> checkpoints/ext_best.keras
```

The Kaggle part is a manual download: extract `data.rar` into
`data/downloads/xainano/` and the script finds it automatically.

**Where results go**

- `models/experiments/tuning_log.csv` – one row per tuning run
- `models/experiments/tuning/` – curves and a summary for every run
- `models/experiments/experiment_log.csv` – the Sprint 2 model runs
- `models/experiments/tuning_table.md`, `extension_table.md` – digit and extension runs, kept separate
- `models/checkpoints/cnn_best.keras`, `ext_best.keras` – the frozen models, each with a
  `.labels.json` (class names) and `.source.json` (which run it came from)
- Per-run tuning models in `models/checkpoints/tuning/` are not committed (too large)

**Two accuracy numbers are reported for every run**

- *Test accuracy* – the normal MNIST test set.
- *Shifted-test accuracy* – the same test set, slightly rotated / shifted / zoomed.
  This is closer to real handwriting coming out of segmentation, and it is where
  augmentation shows its value.

---

## 7. Using the pieces from your own code

**Preprocess + segment + recognise one image**

```python
import sys, json
import cv2
import tensorflow as tf
sys.path.insert(0, "src")                  # run from the repo root
from segmentation import segment_digits    # uses SELECTED_METHOD ("contours")
from preprocessing import preprocess       # uses SELECTED_CONFIG ("otsu_mnist_box")

image = cv2.imread("data/custom_samples/numbers/896_t1.jpg")
model = tf.keras.models.load_model("models/checkpoints/cnn_lenet.keras")

crops, boxes = segment_digits(image)       # boxes are ordered left to right
digits = []
for (x, y, w, h) in boxes:
    crop = image[y:y + h, x:x + w]         # crop the ORIGINAL photo, not the binary crop
    x_in = preprocess(crop, add_channel=True)        # shape (28, 28, 1)
    probs = model.predict(x_in[None], verbose=0)[0]
    digits.append(str(probs.argmax()))
number = "".join(digits)
```

Why crop the original photo: the preprocessing expects the image as it was
taken (dark ink on light paper). `experiments/evaluate_custom.py` (`read_number`)
is a full working example.

**Models trained by the new trainer** come with a label file next to them,
e.g. `cnn_best.keras` + `cnn_best.labels.json`. Read the class names from that
file instead of assuming index = digit. This matters for the extension, where
classes 10–15 are `+ - * / ( )`.

```python
labels = json.load(open("models/checkpoints/cnn_best.labels.json"))["label_names"]
```

---

## 8. Hand-offs and known gaps

- **GUI (Russell):** `gui.py` has the full layout, but `train_model` and
  `predict_digits` are still placeholders. Swap in `segment_digits`,
  `preprocess` and a loaded model as shown in section 7.
- **Which model to load:** `cnn_best.keras` is the frozen digit model. `demo_one.py`
  and `evaluate_custom.py` still load `cnn_lenet.keras` and should be pointed at it.
- **Extension – segmentation:** `segmentation.py` drops any box shorter than 15 %
  of the image height (`min_height_ratio=0.15`). That also drops a minus sign and
  the dots of ÷, and ÷ splits into three pieces. Needs a fix before the
  extension can work.
- **Extension – model (Russell's parser):** `ext_best.keras` outputs 16 classes. Read
  the class names from `ext_best.labels.json`; classes 10–15 are `+ - * / ( )`,
  where `*` covers both × and the asterisk, and `/` means ÷.

---

## 9. Two MNIST loaders – which one to use

- `models/cnn/mnist_loader.py` gives arrays with a fixed train / validation / test
  split. **Use it for training models.**
- `src/prepare_data.py` writes individual digit PNG files to disk. It exists
  because the spec requires building a number image from *a folder of digit
  images*, so the files must physically exist.

---

## 10. Git workflow

Never commit directly to `main`.

```bash
git switch main
git pull
git switch -c feature/<your-name>-<what>    # e.g. feature/liam-eval-harness

# ...work...

git status                                  # check nothing unexpected is listed
git add <files>
git commit -m "sprint N: what you did"
git push -u origin feature/<your-name>-<what>
```

Then open a pull request on GitHub and ask a teammate to review it.

**Rules**

- Every training run gets a row in `models/experiments/experiment_log.csv`
  (the trainer does this automatically for `tuning_log.csv`).
- Do not commit datasets, `venv/`, or per-run tuning models – `.gitignore` covers these.
- The small CSVs and figures in `reports/` **are** committed on purpose: they are
  the evidence of technique comparison the marking scheme asks for.
