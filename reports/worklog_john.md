# Individual Worklog — John (Person A, Data & Vision)

COS30018 Option B — Handwritten Number Recognition System
Owned components: Task 1 (Image Preprocessing), Task 2 (Image Segmentation), automatic image acquisition, custom test set.

---

## Week 3 — w/c 17 Aug 2026

**Sprint:** 1 (Foundations)

### Work completed

**Environment**
- Set up Python virtual environment and installed the project dependency stack (numpy, OpenCV, scikit-image, scikit-learn, matplotlib, pandas, TensorFlow, pytest) on macOS / Python 3.12.9.
- Verified every required package imports correctly via `src/verify_setup.py`.
- Downloaded MNIST (60,000 training images) and exported 500 single-digit PNGs into `data/raw/`, organised one folder per digit.

**Task 1 — Image Preprocessing**
- Implemented seven preprocessing operations as independent, individually testable functions: grayscale conversion, median denoising, Otsu binarization, adaptive thresholding, automatic inversion, centre-of-mass alignment, and normalisation.
- Grouped these into five named configurations so different combinations can be compared rather than assumed.
- Built `experiments/compare_preprocessing.py`, which evaluates every configuration using an identical k-NN classifier on identical MNIST subsets (4,000 train / 1,000 test), so that any accuracy difference is attributable to preprocessing alone.

**Task 2 — Image Segmentation**
- Implemented two segmentation techniques for comparison: contour detection with bounding boxes, and connected-component labelling.
- Both order digits left-to-right by x coordinate and pad each crop to a square before resizing, so narrow digits such as '1' are not horizontally distorted into resembling a '7'.
- Built `experiments/demo_segmentation.py` to measure how often each method recovers the correct digit count, saving an annotated image for every test case.

**Image acquisition**
- Implemented `src/generate_number.py`, which automatically composes multi-digit number images from the folder of individual digit images and records ground-truth labels to `labels.csv`. This satisfies the specification's first required input mode and gives Person C (Liam) labelled data for end-to-end accuracy measurement in Task 4.

**Testing**
- Wrote 36 unit tests covering both modules, including edge cases: empty images, blank inputs, noise specks, unknown configuration names, and aspect-ratio preservation. All 36 pass.

### Results

Preprocessing configuration comparison (k-NN, k=3, 1,000 test images):

| Configuration | Test accuracy | ms/image |
|---|---|---|
| grayscale_only | 0.902 | 0.007 |
| adaptive | 0.901 | 0.015 |
| otsu_denoised | 0.894 | 0.012 |
| otsu_denoised_centered | 0.888 | 0.016 |
| otsu | 0.886 | 0.008 |

Segmentation comparison (20 generated multi-digit images):

| Method | Correct digit count | Accuracy |
|---|---|---|
| Contour detection | 20 / 20 | 100% |
| Connected components | 20 / 20 | 100% |

### Analysis and reflection

The simplest configuration performed best on MNIST, which was not the result I expected. Two explanations account for it. First, binarization discards the grayscale intensity information at stroke edges; because k-NN compares raw pixel distances, those soft anti-aliased edges carry genuine discriminative signal, so throwing them away costs accuracy. Second, centre-of-mass alignment produced no benefit because MNIST digits are *already* centred by mass as part of the dataset's original normalisation — re-centring only introduces small additional shifts, which is consistent with that configuration scoring below plain denoising.

An important caveat limits how far this result can be taken. With 1,000 test images the standard error is approximately ±1 percentage point, so the 0.1-point gap between `grayscale_only` and `adaptive` is indistinguishable from noise; only the ~1.6-point gap between the leading configurations and plain Otsu is meaningful. The honest conclusion is that grayscale and adaptive thresholding are equivalent within error, and both modestly outperform plain Otsu.

More importantly, MNIST is clean, evenly-lit, pre-normalised laboratory data — precisely the conditions under which thresholding and denoising have least to offer. Real photographed handwriting has uneven illumination, shadows, varying pen pressure and paper texture, which is exactly what adaptive thresholding and median denoising exist to handle. I therefore do not treat this as a final technique selection. The comparison must be repeated on a set of genuinely handwritten, photographed images before the project commits to a configuration, and I expect the ranking may change.

### Decision: selected preprocessing technique

Recorded the selection required by Task 1 as a single named constant, `SELECTED_CONFIG` in `src/preprocessing.py`, and set it to `grayscale_only` — the configuration with the highest measured accuracy.

This corrected an inconsistency I found while reviewing the module before pushing: `preprocess()` had been defaulting to `otsu_denoised_centered`, which my own comparison ranked fourth of five. The code was therefore not implementing the technique the evidence supported. Two unit tests now guard against that drifting again: one checks the selected configuration actually exists, the other checks that calling `preprocess()` with no arguments uses it.

The selection is explicitly provisional. It rests on a MNIST-only comparison, and MNIST is the case least favourable to thresholding and denoising. I will re-run the comparison on photographed handwriting and change the selection if adaptive thresholding wins there.

### Matching MNIST's own geometry (20x20 box)

Two people independently pointed at the same gap: Thien in his Sprint 1 report, and the tutor in the Week 8 progress check. Task 1 in the brief says the job is "resizing the image to match with the format recognisable by your ML model", and my resize did not match it.

MNIST was not built by resizing each image to 28x28. Each digit was size-normalised to fit inside a 20x20 box with its aspect ratio kept, then placed in a 28x28 frame with its centre of mass in the middle. A plain resize keeps whatever proportion of the frame the digit happened to occupy in the source photo, so the same digit photographed closer or further away arrives at the model as a different size.

Added `fit_to_mnist_box()` to `src/preprocessing.py`, which crops to the ink, scales the longest side to 20 pixels, pastes into a 28x28 frame and recentres by mass. `reports/mnist_box_comparison.png` shows the effect: the same '7' framed three different ways comes out of a plain resize at 9x12, 16x21 and 7x7 pixels of ink, and out of the new path at 15x20, 15x20 and 14x20. Consistent geometry regardless of how the photo was framed.

Exposed it as two new configurations, `grayscale_mnist_box` and `adaptive_mnist_box`, rather than silently changing the existing ones, so the comparison experiment can measure whether it is actually worth anything instead of me assuming it. Those two have not been scored yet.

Also added an `add_channel` option to `preprocess()`. Thien's Conv2D models take `(28, 28, 1)` and my pipeline was returning `(28, 28)`; the experiment script had been reshaping around it, which works there but would have been a trap for Russell when he wires the GUI to the model.

Six new unit tests cover this, including one that asserts the ink never exceeds 20 pixels on either side and one that a tall thin stroke stays taller than it is wide. 48 tests total, all passing.

### Task 2: four segmentation techniques, and a test set that can tell them apart

The tutor asked for four techniques per part. Task 1 already had seven preprocessing configurations; Task 2 had two, which was the thin side of my own work.

Before adding anything I looked at why the existing comparison was useless: both methods scored 20/20, so it could not justify a choice. The reason was the test data. `generate_number.py` spaced digits 14 pixels apart and never let them touch, which is the easy case for every method. I added a difficulty setting with three levels and measured where the boundary actually is: MNIST tiles carry about 4 blank pixels of margin per side, so spacing has to pass roughly -10 before the ink genuinely merges. The levels are wide (+14), tight (-6, margins overlap but strokes stay separate) and touching (-14, one connected blob).

Then added two techniques:

**Vertical projection profile.** Counts ink per column and cuts at the valleys. It ignores connectivity entirely, so two digits joined by a thin stroke can still be split at the waist. The `valley_ratio` threshold was set by sweeping it against the generated set rather than guessed: 0.05 finds only true gaps and scores 20% on touching digits, 0.40 starts cutting single digits in half, 0.20 is the best compromise at 83% overall.

**Watershed.** Added specifically to attack the touching-digit case, using a distance transform to seed one marker per digit. It did not work, and the reason is worth keeping in the report: the distance transform assumes blob-like objects with a peak in the middle, and digits are strokes of roughly even width, so there is no per-digit peak to seed from. No threshold separates digits without also fragmenting individual strokes. Investigated, measured, rejected with a reason.

Results on 30 generated numbers (`reports/segmentation_comparison.csv`, figure in `reports/segmentation_methods.png`):

| Method | Overall | wide | tight | touching |
|---|---|---|---|---|
| projection | 83% | 90% | 100% | 60% |
| contours | 70% | 90% | 80% | 40% |
| connected components | 70% | 90% | 80% | 40% |
| watershed | 50% | 50% | 60% | 40% |

Three findings the old 20/20 table had hidden. Contours and connected components score identically at every level, because they are not two independent techniques: both find connected regions of ink and differ only in how OpenCV computes them. Projection wins because it is the only one of the four that does not rely on connectivity. And touching digits remain unsolved at 60% for the best method, which is a limitation to state plainly rather than a problem I have fixed.

Recorded the choice as `SELECTED_METHOD = "projection"` and made it the default for `segment_digits()`, with the same two guard tests I added for Task 1. `segment_digits()` had been defaulting to contours, which was the same drift between the written selection and the code that I found in preprocessing last sprint.

69 tests, all passing.

### Task 2: four model-based techniques as well

The tutor's point was that segmentation should also be attacked with models, not only with image processing, and that the same model families used for digit recognition can be reused here. That is a real distinction and it turned out to matter more than I expected.

Every one of the four classical methods decides where a digit ends by looking at pixels, and all four therefore share one ceiling: if two digits' ink touches, no amount of tuning separates them, because to a connectivity-based method they are one object. A model-based method is not bound by that, because it can ask "does this window look like a digit" rather than "are these pixels joined".

Built `src/digit_classifier.py` first, so the segmentation code can swap models without changing. It exposes one interface over two backends: a scikit-learn MLP trained here from `data/raw`, and Thien's LeNet loaded from `models/checkpoints/cnn_lenet.keras`. The MLP has an eleventh class for "not a digit", trained on three kinds of negative patch - blank paper, the join between two digits, and a digit sliced down the middle. Without that class a sliding window labels empty paper as a confident '1'.

Then `src/segmentation_ml.py` with four methods:

- **sliding_mlp** - slide a window at four widths across the ink, score every position, suppress overlapping detections.
- **sliding_cnn** - identical search, Thien's LeNet instead of the MLP. This is the comparison that isolates the model from the search strategy.
- **cutpoint_mlp** - over-segment at every local minimum of the ink profile, then a dynamic program picks the sequence of cuts the model is most confident about. Image processing proposes, the model disposes. This is the classical approach for touching handwriting used in cheque readers.
- **confidence_split** - start from connected blobs, and for any blob far wider than it is tall, try every cut column and keep the one that leaves the model most confident about both halves.

Results on the same 30 images, final run on my own machine (`reports/segmentation_comparison.csv`), with all ten methods including `sliding_cnn`:

| Method | Family | Overall | wide | tight | touching |
|---|---|---|---|---|---|
| projection | classical | 83% | 90% | 100% | 60% |
| confidence_split | model | 80% | 90% | 90% | 60% |
| contours | classical | 70% | 90% | 80% | 40% |
| connected components | classical | 70% | 90% | 80% | 40% |
| cutpoint_mlp | model | 60% | 40% | 80% | 60% |
| sliding_mlp | model | 57% | 60% | 70% | 40% |
| watershed | classical | 50% | 50% | 60% | 40% |
| sliding_cnn | model | 47% | 30% | 60% | 50% |
| sliding_svm | model | 37% | 40% | 60% | 10% |
| sliding_rf | model | 23% | 40% | 20% | 10% |

**Correction to an earlier draft of this entry.** A first run had `cutpoint_mlp` at 90% on touching digits and I wrote that up as the headline. That number was measured before the `to_patch()` fix described below, when the classifiers were scoring patches in a different format from the one they were trained on. After the fix it is 60%, the same as projection. The honest result is that none of the ten methods gets past 60% on touching digits: the model-based family does not break the ceiling I had documented, it only matches it. The figure `reports/segmentation_touching.png` from that first run was also out of date; both Task 2 figures are now produced by `experiments/make_segmentation_figures.py` so they always match the current code.

What survives is a different reading of the table. The two families fail in different places: `cutpoint_mlp` is strong on tight and touching digits but the worst on well-separated ones (40%), because it over-segments when there is nothing to resolve, while the classical methods are the reverse. `confidence_split` runs the cheap classical step first and only asks the model where the blob geometry says a decision is needed, and it comes second overall at a fraction of the sliding window's cost.

### Four distinct models behind the sliding window

The tutor's requirement was four *different models*, not four uses of one, so `digit_classifier.py` has four backends from four different families: a fully-connected network (MLP), a convolutional network (Thien's LeNet), an RBF kernel machine (SVM) and an ensemble of decision trees (random forest). The CNN is deliberately the same architecture Thien uses for Task 3. All four are driven by the same sliding-window search, so anything that differs between them is the model and nothing else.

Getting that comparison to be fair exposed two bugs worth recording.

The window filter used an absolute confidence threshold of 0.55, which suited the MLP and silently returned **nothing at all** for the SVM and the forest. Both scored 0%, and I nearly wrote that up as "these models cannot segment". They can: a softmax peaks near 1.0, while Platt scaling and vote-averaging across 300 trees both top out near 0.5, so no window ever cleared the bar. The threshold is now relative - a window is kept if it scores at least half of the best window in the same image - which removes the calibration difference instead of tuning around it.

The second was a train/test mismatch. The sliding window padded each crop to a square before resizing, but the training tiles were resized directly, so every model was scoring patches slightly outside the distribution it was fitted on. Both paths now go through one shared `to_patch()` function.

The most useful finding is in the `sliding_cnn` row. Thien's LeNet recognises single MNIST digits at 99%, far better than my MLP, yet it segments worse: 47% against 57%. The reason is that my MLP was trained with an eleventh "not a digit" class and the CNN was not. The CNN was only ever shown clean, centred digits, so when a window lands on half of one digit and half of the next it still has to pick one of ten answers, and I can only infer "not a digit" indirectly from how flat its softmax is. For sliding-window segmentation, being able to say "no" matters more than raw recognition accuracy. The CNN is still the best of the four on touching digits (50%), where its learned features help.

Two caveats. The three scikit-learn models are trained on the 500 exported digit images, not the full 60,000, so they are handicapped and the ordering may change when retrained. And every number here comes from generated images; the real handwritten set is still to come.

96 tests, all passing.

### Sprint 3 - preparing for the real handwritten set

Before photographing real handwriting I tested the preprocessing on a synthetic phone-style photo: off-white textured paper with a shading gradient. Two faults showed up that clean MNIST-style tests could never have caught.

- **The 20x20 box did nothing on a photo.** `fit_to_mnist_box()` located the digit as "every non-zero pixel". Inverted paper is never exactly 0, so every pixel counted and the crop became the whole 800x600 frame; the digit was shrunk together with all the paper around it instead of filling the 20-pixel box. It now locates the ink with an Otsu mask and only uses the mask to decide where to crop, keeping the original grey values. Same test image: 15x20.
- **Adaptive thresholding hollowed out thick strokes.** The block size was fixed at 11 pixels. On a large photo a pen stroke is wider than that, so the inside of the stroke is compared only against itself and comes out as paper; only 26% of the stroke survived. The block now scales with the image (an eighth of the shorter side, never below 11), which recovers the whole stroke. On 28x28 MNIST it is still 11, so the existing comparison numbers are unaffected. The constant `c=2` also marked about 30% of the noisy paper as ink, where `c=10` gave almost none; I have left it at 2 until the real set shows what real paper does.

- **Tight digit crops were fed to the model the wrong way round.** Wiring segmentation to the classifier for the first time, I found that a tight box around a thick digit can be more than half ink. `invert_if_dark_strokes()` decided on the image's overall mean, so those crops stayed black-on-white and the model saw the opposite of what it was trained on. It now judges the background from the image border, which is paper however much ink sits inside. On MNIST-derived images the decision is identical, so earlier numbers still stand.

All three fixes have regression tests.

I also wrote `experiments/evaluate_custom.py`, ready for the photos. It repeats the Task 1 comparison on real single digits, and for real numbers it runs segmentation followed by recognition of every crop, reporting both "found the right number of digits" and "read the whole number exactly". A smoke test on synthetic photo-style numbers already suggested something worth checking on the real set: with plain resizing a tight crop is stretched to fill 28x28, while the 20x20 box keeps its shape, so `SELECTED_CONFIG` may need to change for the end-to-end pipeline even though `grayscale_only` won on MNIST. The synthetic smoke test is not evidence either way; the real photos will decide.

### Sprint 3 - first results on real handwriting

**Test set.** I wrote the test set by hand on two A4 sheets and photographed them with a phone. `src/split_sheet.py` cuts a sheet into one image per number, keeps a margin of paper around each, and marks a number as touching only when it measurably has fewer ink blobs than digits, rather than when I intended it to touch. The set has 70 single digits (7 per digit, written in six deliberately different styles: small, large, slanted, fast, uneven height) and 27 multi-digit numbers of 2 to 5 digits, 8 of which touch. Both original photos are kept in `data/custom_samples/sheets/`.

**MNIST first, repeated properly.** Two single runs of the MNIST comparison ranked the configurations differently, so `compare_preprocessing.py` now trains each configuration five times with different seeds. With that, the top three are tied: otsu 0.967 +/- 0.006, grayscale_only 0.967 +/- 0.005, adaptive 0.967 +/- 0.003. MNIST cannot separate them.

**Real photos separate them immediately.** Same team CNN, 70 real digits (`reports/custom_preprocessing.csv`):

| Configuration | MNIST (5 seeds) | Real photos |
|---|---|---|
| otsu | 0.967 | **82.9%** |
| otsu_denoised | 0.959 | 82.9% |
| otsu_denoised_centered | 0.956 | 80.0% |
| adaptive | 0.967 | 51.4% |
| grayscale_only (was selected) | 0.967 | 32.9% |
| grayscale_mnist_box | 0.962 | 28.6% |
| adaptive_mnist_box | 0.965 | 20.0% |

The configuration I had selected on MNIST reads one real digit in three; Otsu reads more than four in five. The reason is the background. MNIST's background is exactly 0, and a photo's is not: after inversion, grayscale_only hands the CNN a digit on a grey haze it has never seen, while Otsu forces the paper to exactly 0 and the input looks like MNIST again. This is the caveat I recorded in Sprint 2 - that MNIST is the case least favourable to thresholding - measured rather than assumed, and larger than I expected: 50 percentage points. With 70 images the standard error is about 5 points, so the Otsu family's lead is far outside noise; the gaps within the Otsu family are not.

The 20x20 box did not help on single digits, which surprised me. These crops already carry a margin of paper, so a plain resize leaves the digit at roughly MNIST scale anyway, and the box's own ink-finding step is thrown off by adaptive thresholding's speckle on real paper. The box should matter for the tight crops segmentation produces, so I added `otsu_mnist_box` and a whole-number test across every method and configuration (`evaluate_custom.py --all-configs`) to settle the combination on end-to-end results.

**Segmentation: the Sprint 2 selection failed.** Counting digits correctly on the 27 real numbers (`reports/custom_segmentation.csv`):

| Method | Generated | Real photos |
|---|---|---|
| contours | 70% | **74%** |
| connected components | 70% | 74% |
| watershed | 50% | 52% |
| projection (was selected) | 83% | **19%** |

Projection was the best method on generated data and the worst on real handwriting. A real pen stroke is thin, so a column through the middle of a 0 or a 6 holds only a few pixels of ink, and the valley threshold I had tuned on MNIST-thick strokes reads that as a gap between digits; "60" came out as eleven pieces. `SELECTED_METHOD` is now `contours`. Touching digits remain unsolved: 2 of the 8 touching numbers are counted correctly.

The lesson I take from this sprint is concrete: both of my Sprint 2 selections were made on data that could not tell the options apart, and both were wrong. The comparison only became informative once it ran on the kind of input the system will actually receive.

### Sprint 3 - the final selection, chosen end to end

Segmentation and preprocessing interact, so I chose them together: every segmentation method against every preprocessing configuration, on the 27 real numbers, counting a number as right only if every digit is right (`evaluate_custom.py --all-configs`, `reports/custom_pipeline.csv`). With contours as the segmenter:

| Configuration | Real single digits | Real numbers, end to end |
|---|---|---|
| **otsu_mnist_box** (selected) | 82.9% | **63%** |
| adaptive_mnist_box | 20.0% | 56% |
| grayscale_mnist_box | 28.6% | 19% |
| otsu | 82.9% | 7% |
| grayscale_only (Sprint 2 choice) | 32.9% | 0% |

This is the result that justifies the 20x20 box. On single digits it made no difference to Otsu at all - both read 58 of 70 - because those crops already carry a margin of paper. End to end it is the whole difference: segmentation hands over tight crops, a plain resize stretches each one to fill the frame, and Otsu without the box falls from 83% to 7%. With the box it reads 63% of whole numbers, and 90% of individual digits in the numbers that were segmented correctly.

`SELECTED_CONFIG` is now `otsu_mnist_box` and `SELECTED_METHOD` is `contours`. It is the only configuration at the top of both real tests. I would not claim it beats `adaptive_mnist_box` on the end-to-end figure alone - 63% against 56% on 27 numbers is within the roughly 9-point standard error - but on single digits it is 83% against 20%, which is not.

The remaining error is now mostly segmentation, not recognition. Of the 10 numbers read wrongly, 7 were split into the wrong number of pieces - 6 because the digits touch, one because the top bar of a 5 came away as a separate blob - and only 3 were segmented correctly but misread. Touching digits are the limit of the whole system, not only of Task 2.

### Blockers / risks

- Touching digits: no method passes 60%. To be documented as a limitation and re-tested on real handwriting.
- The two `mnist_box` configurations have not been scored yet; that needs a run of `compare_preprocessing.py` on my machine, where TensorFlow is installed.
- None of my code is on `main` yet, which blocks Russell's pipeline and Liam's evaluation. Opening the pull request is the first Sprint 3 action.
- 70 real digits is enough to separate the Otsu family from the rest, not to rank within it.
- One writer, one pen, one paper. The next step is a sheet from each teammate, and one sheet with a black pen, lined paper and a side light.

### Next week

- Open the pull request into `main` and send Russell and Liam the entry points; `segment_digits()` and `preprocess()` now default to the selected pair, so the pipeline needs no extra settings.
- Re-run `compare_preprocessing.py` so `otsu_mnist_box` has its MNIST figure next to the others.
- Collect a sheet of handwriting from each teammate, plus one with a different pen, lined paper and uneven light, and re-run both real-data comparisons on the larger set.
- Draft the Data preprocessing and Image segmentation report sections from these tables.
