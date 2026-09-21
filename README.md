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
| `src/preprocessing.py` | John | Task 1 — grayscale, denoise, Otsu / adaptive threshold, centring, normalisation, exposed as five named configurations for comparison |
| `src/segmentation.py` | John | Task 2 — contour and connected-component segmentation into ordered single-digit crops |
| `src/prepare_data.py` | John | Exports MNIST digits as PNGs into `data/raw/` |
| `src/generate_number.py` | John | Builds multi-digit number images + ground-truth labels (automatic image acquisition) |
| `src/show_progress.py` | John | Prints a status summary of all deliverables |
| `experiments/` | John | Scripts producing the Task 1 and Task 2 comparison evidence |
| `models/cnn/` | Thien | CNN architectures (shallow, LeNet, VGG-small), shared MNIST loader, experiment logger |
| `models/experiments/` | Thien | Training histories and `experiment_log.csv` |
| `gui.py` | Russell | GUI — image input and display |
| `tests/` | John | 36 unit tests for preprocessing and segmentation |
| `reports/` | John | Comparison CSVs, figures, individual worklog |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/verify_setup.py
```

## Running

| What | Command |
|---|---|
| Verify environment | `python src/verify_setup.py` |
| Export MNIST digit images | `python src/prepare_data.py` |
| Generate multi-digit test images | `python src/generate_number.py` |
| Task 1 comparison experiment | `python experiments/compare_preprocessing.py` |
| Task 2 segmentation comparison | `python experiments/demo_segmentation.py` |
| Unit tests | `python -m pytest tests/ -v` |
| Progress summary | `python src/show_progress.py` |

## Two MNIST loaders — which to use

Both exist on purpose:

- `models/cnn/mnist_loader.py` (Thien) returns normalised arrays with a train/val/test split — use this for **training models**.
- `src/prepare_data.py` (John) writes individual digit PNGs to disk — this exists because the specification requires the system to build a number image from *a folder of digit images*, so the files have to physically exist.

## Conventions

- Branch → commit → pull request → review → merge. No direct commits to `main`.
- Branch naming: `feature/<name>-<what>`.
- Every training run gets a row in `models/experiments/experiment_log.csv`.
- Datasets and virtual environments are gitignored; the small comparison CSVs and figures in `reports/` are committed on purpose, because they are the evidence of technique comparison the marking scheme asks for.
