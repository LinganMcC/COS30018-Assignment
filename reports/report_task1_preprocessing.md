# Data Preprocessing (Task 1)

*Owner: John. Report section for "System descriptions → Data preprocessing".*

## 1. What this stage has to do

Preprocessing sits between the camera and the classifier. Its contract is
fixed by the models: whatever arrives — a phone photo of a page, a crop handed
over by segmentation, or an MNIST array — it must leave as a 28×28 `float32`
array in `[0, 1]`, white strokes on a black background, in the same geometry
MNIST itself uses.

That last clause is the part that is easy to miss. The classifiers are trained
on MNIST, so MNIST is not just the training data, it is the specification of
the input format. Anything that makes a real photo *look* less like an MNIST
digit costs accuracy at inference time, however reasonable it looks to a human
eye.

The implementation is `src/preprocessing.py`.

## 2. Techniques investigated

Rather than compare whole pipelines as black boxes, preprocessing was built out
of six primitive operations that can be switched on and off independently:

| Operation | Function | What it does |
|---|---|---|
| Grayscale | `to_grayscale` | Drops colour, which carries no information about digit identity |
| Denoise | `denoise` | 3×3 median blur, to remove paper speckle and sensor noise |
| Otsu binarization | `binarize_otsu` | One global threshold chosen to best separate two brightness peaks |
| Adaptive binarization | `binarize_adaptive` | A threshold computed per neighbourhood, for uneven lighting |
| Centre of mass | `center_by_mass` | Shifts the digit so its ink centroid is at the frame centre |
| MNIST 20×20 box | `fit_to_mnist_box` | Scales the digit into a 20×20 box inside a 28×28 frame, aspect preserved |

Eight named configurations were then assembled from these and compared
(`CONFIGS` in `src/preprocessing.py`):

| Configuration | Threshold | Denoise | Geometry |
|---|---|---|---|
| `grayscale_only` | none | no | plain resize |
| `otsu` | Otsu | no | plain resize |
| `otsu_denoised` | Otsu | yes | plain resize |
| `otsu_denoised_centered` | Otsu | yes | plain resize + centre of mass |
| `adaptive` | adaptive | yes | plain resize |
| `grayscale_mnist_box` | none | no | 20×20 MNIST box |
| `adaptive_mnist_box` | adaptive | yes | 20×20 MNIST box |
| `otsu_mnist_box` | Otsu | no | 20×20 MNIST box |

The two axes under test are therefore **how the ink is separated from the
paper** (nothing / Otsu / adaptive) and **how the digit is placed in the frame**
(plain resize / MNIST's own 20×20 convention).

## 3. How they were compared

One measurement was not enough, and finding that out was itself a result. Three
were used, each answering a different question.

**(a) MNIST, five seeds.** Each configuration was applied to the MNIST test set
and classified by the team LeNet, repeated over five random seeds so that
run-to-run variation could be separated from a real difference.
Script: `experiments/compare_preprocessing.py` → `reports/preprocessing_comparison.csv`.

**(b) 149 real single-digit photos.** Fourteen or fifteen handwritten samples
of each digit 0–9, from six photographed sheets written by four people — all
four team members — one of them deliberately on hand-ruled lines with slanted
writing. This is the first test where thresholding has anything to do, because
real paper is not uniformly white.
Script: `experiments/evaluate_custom.py` → `reports/custom_preprocessing.csv`.
Who wrote which sheet: `data/custom_samples/sheets.csv`.

**(c) 99 real multi-digit numbers, read end to end.** The whole pipeline —
segment, crop, preprocess, classify, join — with a number counted correct only
if every digit in it is correct. This is the only test that sees the crops
segmentation actually produces, which are much tighter than a hand-framed photo
of a single digit.
Script: `evaluate_custom.py --pipeline-only` → `reports/custom_pipeline.csv`.

## 4. Results

| Configuration | (a) MNIST, 5 seeds | (b) 149 real digits | (c) 99 real numbers |
|---|---|---|---|
| **`otsu_mnist_box`** | 0.960 ± 0.005 | **81.9 %** | **62 %** |
| `otsu_denoised_centered` | 0.956 ± 0.005 | 79.9 % | 7 % |
| `otsu` | 0.967 ± 0.006 | 76.5 % | 1 % |
| `otsu_denoised` | 0.959 ± 0.005 | 76.5 % | 1 % |
| `adaptive` | 0.967 ± 0.003 | 63.1 % | 2 % |
| `grayscale_only` | 0.967 ± 0.005 | 61.1 % | 1 % |
| `grayscale_mnist_box` | 0.962 ± 0.003 | 38.3 % | 11 % |
| `adaptive_mnist_box` | 0.965 ± 0.003 | 33.6 % | 34 % |

Column (c) uses `contours` segmentation (the Task 2 selection) for every row, so
only the preprocessing changes. An earlier version of this table was measured on
70 digits and 27 numbers from one writer; the ranking at the top did not change
when the set grew to four writers.

Supporting figures: `reports/preprocessing_visual.png` (MNIST digits through
each configuration) and `reports/preprocessing_real_visual.png` (the same for a
photographed digit).

### What each measurement showed

**MNIST cannot choose.** All eight configurations land between 0.956 and 0.967,
a spread comparable to the ±0.005 seed-to-seed standard deviation. This is not a
flaw in the experiment, it is a property of the dataset: MNIST digits are
already clean, already centred, and already inside a 20×20 box, so denoising has
no noise to remove and the box operation has nothing left to do. Had the project
stopped at this table it would have had no evidence at all for its choice.

**On real photos, thresholding is what matters.** The spread opens from one
percentage point to almost fifty. Otsu forces the paper to exactly 0, which is
what MNIST's background is; `grayscale_only` leaves a grey haze across every
pixel that the classifier has never seen in training. Adaptive thresholding sits
in between — it handles uneven lighting well but, with a neighbourhood scaled to
the image, it also finds "edges" inside the broad flat interior of a thick
stroke and hollows it out.

**End to end, the geometry is what matters.** `otsu` and `otsu_mnist_box` differ
only in how the digit is placed in the 28×28 frame, and they score 1 % and 62 %
on whole numbers against 76.5 % and 81.9 % on single digits. The reason is that
segmentation hands over a *tight* crop with almost no margin. A plain resize
stretches that crop to fill the full 28×28 frame, producing a digit far larger
and differently proportioned than any MNIST training sample. `fit_to_mnist_box`
reproduces the dataset's own convention instead, and the loss disappears.

The same effect explains the one apparent anomaly in the table:
`adaptive_mnist_box` is the worst on single digits (33.6 %) yet second-best end
to end (34 %). The box is worth so much on tight crops that it partly
compensates for a weaker threshold.

### Results by writer

The selected pipeline on each sheet (`evaluate_custom.py --by-sheet-only` →
`reports/custom_results_by_image.csv`):

| Sheet | Writer | Single digits | Numbers | Read exactly |
|---|---|---|---|---|
| IMG_1406 | John | 90 % | 21 | 57 % |
| IMG_1407 | John | 83 % | 6 | 83 % |
| IMG_2005 | Thien | 89 % | 18 | 67 % |
| IMG_2006 | Liam | 100 % | 18 | 72 % |
| IMG_2007 | Russell | 100 % | 18 | 72 % |
| IMG_2010 | John, ruled lines + slant | **30 %** | 18 | 33 % |

On the five plain sheets single digits are read 90 % of the time (116 of 129),
with no writer below 83 %, so the selection is not tuned to one person's hand.
The ruled sheet is the exception and is explained in the limitations below.

## 5. Selected technique

**`otsu_mnist_box`** — grayscale → Otsu binarization → fit to a 20×20 box inside
a 28×28 frame, centred by mass → normalise to `[0, 1]`.

It is the only configuration at the top of both real-data tests. On single
digits alone its lead is not decisive: `otsu_denoised_centered` is 3 images
behind out of 149, which is within noise. Measurement (c) decides it — the same
two configurations read 62 % and 7 % of whole numbers, because only one of them
keeps a tight crop in MNIST's proportions.

Denoising was dropped from the selection. It is implemented and available, but
on this test set it never helped: median blur also erodes the thin end of a pen
stroke, and the photographs were clean enough that it cost more than it gained.

The choice is recorded as `SELECTED_CONFIG` in `src/preprocessing.py`, which is
the default argument of `preprocess()`, and a unit test asserts that the default
path and the named selection give identical output — so the technique written up
here cannot silently drift from the technique the system runs.

## 6. Three implementation details that changed the result

These are worth recording because in each case the obvious implementation was
measurably wrong.

**Deciding when to invert.** Strokes have to end up light on dark. The first
implementation judged "is this image mostly dark?" from the mean of the whole
image, which fails on a tight crop: a thick digit can be more than half ink, so
its mean is dark even though the paper is white, and the crop was left
un-inverted. The border of an image is almost always paper however much ink is
inside it, so `invert_if_dark_strokes` now takes the median of the four edges.

**Scaling the adaptive neighbourhood.** A fixed 11-pixel block suits a small
scan, but on an 800×600 photo it is narrower than a pen stroke, so the inside of
a thick stroke is compared only against itself and comes out as paper — about
three quarters of the stroke was lost. The block size now scales with the image
(an eighth of the shorter side, minimum 11, forced odd).

**Finding the ink before boxing it.** `fit_to_mnist_box` has to locate the digit
before it can scale it. Using "any non-zero pixel" works on a clean scan but not
on a photo, where inverted paper is never exactly zero, so every pixel counts as
ink and the bounding box silently becomes the whole frame — turning the box
operation into a plain resize without any error. The ink is now located with an
Otsu mask, while the pixels copied into the box keep their original grey values.

## 7. Limitations

- **Ruled lines are not removed.** On the lined sheet single digits fall to
  30 % (6 of 20). Otsu keeps the printed or drawn line as ink, and the 20×20
  box then fits around digit and line together, so a digit sitting on a line
  reaches the classifier with a bar along its base — 7 of the 14 errors were
  read as "2". The row written above its line scored 5 of 10, the row touching
  its line 1 of 10, so the line rather than the slant is the cause. Removing
  long horizontal strokes before thresholding (a morphological opening, as
  `split_sheet.py` already does to locate numbers) is the obvious next step.
- **The real test set is 149 digits and 99 numbers from four writers.** Pen
  thickness varies by a factor of about 5.8 across it
  (`reports/custom_test_set.csv`), but it is one camera and one paper stock, and
  every sheet copies the same template of numbers. The percentages should be
  read as evidence for ranking the configurations, not as an estimate of
  accuracy in the wild.
- **Otsu assumes two brightness peaks.** On a photo with a strong shadow
  gradient across the page there is no single threshold that works, and
  `adaptive` would be the better choice. The selection is right for the
  conditions tested, not unconditionally.
- **Denoising is available but unused**, so a genuinely noisy input is not
  handled as well as it could be. Switching `SELECTED_CONFIG` to
  `otsu_denoised` is a one-line change if that becomes necessary.
- **Colour is discarded at the first step.** This is correct for ink on paper
  and would not be for a coloured-background source.
