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

Results on the same 30 images (`reports/segmentation_comparison.csv`):

| Method | Family | Overall | wide | tight | touching |
|---|---|---|---|---|---|
| projection | classical | 83% | 90% | 100% | 60% |
| confidence_split | model | 80% | 90% | 90% | 60% |
| contours | classical | 70% | 90% | 80% | 40% |
| connected components | classical | 70% | 90% | 80% | 40% |
| cutpoint_mlp | model | 70% | 40% | 80% | **90%** |
| sliding_mlp | model | 50% | 80% | 50% | 20% |
| watershed | classical | 50% | 50% | 60% | 40% |

The headline is in the last column. Touching digits had been the ceiling on everything I had built: the best classical method reached 60%, and I had written that up as an open limitation. `cutpoint_mlp` reaches 90% on exactly that case. `reports/segmentation_touching.png` shows it on one image - six methods return one or two boxes for a three-digit number, the cut-point classifier returns three.

It is also the worst method on well-separated digits, at 40%, because it over-segments when there is no ambiguity to resolve. So the honest reading is not that one method wins. It is that the two families fail in opposite places, and the best overall result would come from running the cheap classical method first and only calling the model where the blob geometry says a decision is needed - which is roughly what `confidence_split` does, and it is second overall while being far cheaper than the sliding window.

### Four distinct models behind the sliding window

The tutor's requirement was four *different models*, not four uses of one, so I extended `digit_classifier.py` to four backends chosen to be four different families rather than four settings of the same idea: a fully-connected network (MLP), a convolutional network (Thien's LeNet), an RBF kernel machine (SVM) and an ensemble of decision trees (random forest). The CNN is deliberately the same architecture Thien uses for Task 3; sharing it is the point, because if the strongest recogniser also segments best that is worth knowing, and if it does not, the bottleneck is the search rather than the model.

All four are driven by the same sliding-window search, so anything that differs between them is the model and nothing else.

Getting that comparison to be fair exposed a bug worth recording. The window filter used an absolute confidence threshold of 0.55, which suited the MLP and silently returned **nothing at all** for the SVM and the forest - both scored 0%, and I nearly wrote that up as "these models cannot segment". They can. Their probabilities are simply on a different scale: a softmax peaks near 1.0, while Platt scaling and vote-averaging across 300 trees both top out near 0.5, so no window ever cleared the bar. I replaced the absolute threshold with a relative one, keeping windows that score at least half of the best window *in the same image*, which removes the calibration difference instead of tuning around it. SVM went from 0% to 37% and the forest from 0% to 27%.

A second train/test mismatch came out of the same investigation. The sliding window pads each crop to a square before resizing, but the training tiles were resized directly, so every model was scoring patches slightly outside the distribution it was fitted on. Both paths now go through one shared `to_patch()` function. The MLP gained a little from this; the SVM and forest needed it far more, which fits - they are much less tolerant of that kind of drift than a network is.

| Model | Family | Overall | wide | tight | touching |
|---|---|---|---|---|---|
| sliding_mlp | fully-connected network | 57% | 60% | 70% | 40% |
| sliding_svm | RBF kernel machine | 37% | 40% | 60% | 10% |
| sliding_rf | decision-tree ensemble | 27% | 50% | 20% | 10% |
| sliding_cnn | convolutional network | not measured yet | | | |

`sliding_cnn` is written and registered but could not be measured here, because TensorFlow is not installed in the environment I was testing in. It runs on my own machine and the number goes in the report.

All three measurable models sit below the plain classical methods, and I would rather state that than bury it. The likely reason is the training data: these are fitted on the 500 exported digit images, not the full 60,000, so the comparison currently measures a handicapped version of each model. Retraining on full MNIST is the first Sprint 3 item, and I expect the ordering to change.

One caveat on the MLP: it is trained on the 500 exported digit images, not the full 60,000. Its ceiling is lower than it should be, and `sliding_mlp`'s 50% partly reflects that rather than the method. Retraining it on full MNIST is a Sprint 3 item.

83 tests, all passing.

### Blockers / risks

- No blockers on my components.
- Risk noted for later: touching or overlapping digits are a known hard case for contour-based segmentation. Current generated data uses generous spacing, so the 100% result should not be read as robustness to real handwriting. I will document this as a limitation and test it explicitly against the real handwritten set.
- The tutor also asked for at least four models in the Task 3 comparison. The team currently has three, and all three are CNNs, so they are one technique in three configurations rather than the "different techniques" the brief asks for. My k-NN is the obvious fourth and the only non-CNN in the project, but it currently runs on 12,000 training images as a measuring instrument for Task 1, not as a model in its own right. It needs a proper run on the full 60,000 before it can stand next to the CNNs.

### Next week

- Photograph and prepare a real handwritten test set into `data/custom_samples/`.
- Re-run the preprocessing comparison on that set and compare the ranking against the MNIST result.
- Provide the labelled generated dataset to Liam for Task 4 and the number generator to Russell for GUI wiring.
