# COS30018-Assignment

Option B — Handwritten Number Recognition System (HNRS)

Team: John (Data & Vision) · Thien (Deep Models) · Liam (Baselines & Evaluation) · Russell (Systems, Integration & Extension)

---

## Pipeline

```
input image ──► preprocessing ──► segmentation ──► digit recognition ──► number reconstruction ──► output
                (Task 1, John)   (Task 2, John)   (Task 3, Thien/Liam)  (Task 4, Liam)
```

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| `src/preprocessing.py` | John | Task 1 — grayscale, denoise, Otsu / adaptive threshold, centring, the MNIST 20x20 box, normalisation, exposed as seven named configurations; the chosen one is `SELECTED_CONFIG` |
| `src/segmentation.py` | John | Task 2, classical — contours, connected components, projection profile, watershed; the chosen one is `SELECTED_METHOD` |
| `src/segmentation_ml.py` | John | Task 2, model-based — sliding window with four different models, cut-point and confidence-split strategies |
| `src/digit_classifier.py` | John | One interface over four classifiers (MLP, CNN, SVM, random forest) used by the model-based segmenters |
| `src/prepare_data.py` | John | Exports MNIST digits as PNGs into `data/raw/` |
| `src/generate_number.py` | John | Builds multi-digit number images + ground-truth labels (automatic image acquisition) |
| `src/show_progress.py` | John | Prints a status summary of all deliverables |
| `experiments/` | John | Scripts producing the Task 1 and Task 2 comparison evidence, the report figures, and the real-handwriting evaluation |
| `models/cnn/` | Thien | CNN architectures (shallow, LeNet, VGG-small), shared MNIST loader, experiment logger |
| `models/experiments/` | Thien | Training histories and `experiment_log.csv` |
| `gui.py` | Russell | GUI — image input and display |
| `tests/` | John | Unit tests for preprocessing and both segmentation modules |
| `data/custom_samples/` | John | Real handwritten photos: `digits/3_01.jpg`, `numbers/94026_01.jpg` (label before the underscore) |
| `reports/` | John | Comparison CSVs, figures, individual worklog |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/verify_setup.py
```

## Running — from a fresh clone, in this order

Datasets and trained scikit-learn models are gitignored, so a new clone has to rebuild them. Each step needs the one before it.

| # | What | Command |
|---|---|---|
| 1 | Verify environment | `python src/verify_setup.py` |
| 2 | Export MNIST digit images to `data/raw/` | `python src/prepare_data.py` |
| 3 | Generate labelled multi-digit images (wide / tight / touching) | `python src/generate_number.py --count 30` |
| 4 | Train the MLP, SVM and random forest used by the model-based segmenters | `python src/digit_classifier.py` |
| 5 | Unit tests | `python -m pytest tests/ -q` |
| 6 | Task 1 comparison (retrains the team CNN per configuration, slow) | `python experiments/compare_preprocessing.py` |
| 7 | Task 2 comparison, all segmentation methods | `python experiments/demo_segmentation.py` |
| 8 | Task 2 report figures | `python experiments/make_segmentation_figures.py` |
| 9 | Both comparisons on real handwriting in `data/custom_samples/` | `python experiments/evaluate_custom.py` |
| – | Progress summary | `python src/show_progress.py` |

Step 6 needs TensorFlow. Steps 7, 8 and 9 use Thien's CNN where available and skip only the CNN parts if TensorFlow is missing.

## Using Task 1 and Task 2 from other code

```python
from segmentation import segment_digits          # uses SELECTED_METHOD
from preprocessing import preprocess             # uses SELECTED_CONFIG

crops, boxes = segment_digits(image)             # boxes ordered left to right
for (x, y, w, h) in boxes:
    digit = image[y:y + h, x:x + w]              # crop the ORIGINAL image, not the binary crop
    x_in = preprocess(digit, add_channel=True)   # (28, 28, 1), ready for the team CNN
```

Crop the original image with the returned boxes rather than passing the binary crops to `preprocess()`: the preprocessing configurations expect the picture as it was taken, dark ink on light paper. `experiments/evaluate_custom.py` (`read_number`) is a working example of the whole chain.

## Two MNIST loaders — which to use

Both exist on purpose:

- `models/cnn/mnist_loader.py` (Thien) returns normalised arrays with a train/val/test split — use this for **training models**.
- `src/prepare_data.py` (John) writes individual digit PNGs to disk — this exists because the specification requires the system to build a number image from *a folder of digit images*, so the files have to physically exist.

## Conventions

- Branch → commit → pull request → review → merge. No direct commits to `main`.
- Branch naming: `feature/<name>-<what>`.
- Every training run gets a row in `models/experiments/experiment_log.csv`.
- Datasets and virtual environments are gitignored; the small comparison CSVs and figures in `reports/` are committed on purpose, because they are the evidence of technique comparison the marking scheme asks for.
