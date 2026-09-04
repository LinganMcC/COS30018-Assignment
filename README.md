# HNRS — Handwritten Number Recognition System

COS30018 Intelligent Systems — Option B  
Team: John (A · Data & Vision) · Thien (B · Deep Models) · Liam (C · Baselines & Evaluation) · Russell (D · Systems, Integration & Extension)

---

## What this system does

Takes an image containing a handwritten multi-digit number and outputs the recognised number.

```
input image ──► preprocessing ──► segmentation ──► digit recognition ──► number reconstruction ──► output
                (Task 1, John)   (Task 2, John)   (Task 3, Thien/Liam)  (Task 4, Liam)
```

---

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt. Re-run the activate command every time you open a new terminal.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify the setup

```bash
python src/verify_setup.py
```

Expect version numbers for every library and no red error text.

---

## Running John's modules (Tasks 1 & 2)

All commands are run from the repository root with the venv active.

| What | Command | Output |
|---|---|---|
| Verify environment | `python src/verify_setup.py` | Library versions printed |
| Download MNIST + build sample data | `python src/prepare_data.py` | `data/raw/` filled with single-digit PNGs |
| Generate a multi-digit number image | `python src/generate_number.py` | `data/generated/` images + `labels.csv` |
| Task 1 comparison experiment | `python experiments/compare_preprocessing.py` | `reports/preprocessing_comparison.csv` + side-by-side images |
| Task 2 segmentation demo | `python experiments/demo_segmentation.py` | `reports/segmentation_demo/` annotated images |
| Run the tests | `python -m pytest tests/ -v` | All tests pass |

---

## Folder structure

```
hnrs/
├── data/
│   ├── raw/            single-digit images (generated from MNIST)
│   ├── generated/      auto-created multi-digit number images + ground truth
│   └── custom_samples/ real handwritten photos (John collects these)
├── src/
│   ├── preprocessing.py    Task 1 — John
│   ├── segmentation.py     Task 2 — John
│   ├── generate_number.py  image acquisition — John
│   ├── prepare_data.py     dataset helper
│   └── verify_setup.py     environment check
├── experiments/            comparison scripts that produce report evidence
├── reports/                generated figures, CSVs, worklogs
├── tests/                  unit tests
├── saved_models/           trained models (Thien/Liam)
└── requirements.txt
```

---

## Team conventions

- **Never commit to `main` directly.** Branch → commit → pull request → review → merge.
- Branch naming: `feature/preprocessing`, `feature/cnn-model`, `feature/gui`, etc.
- Every experiment run gets a row in `reports/experiment_log.csv`.
- Write your report section in the same sprint you build the thing.
- `venv/`, `data/`, and `saved_models/` are gitignored — do not commit large binaries.
