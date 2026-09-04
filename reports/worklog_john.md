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

### Blockers / risks

- No blockers on my components.
- Risk noted for later: touching or overlapping digits are a known hard case for contour-based segmentation. Current generated data uses generous spacing, so the 100% result should not be read as robustness to real handwriting. I will document this as a limitation and test it explicitly against the real handwritten set.

### Next week

- Photograph and prepare a real handwritten test set into `data/custom_samples/`.
- Re-run the preprocessing comparison on that set and compare the ranking against the MNIST result.
- Provide the labelled generated dataset to Liam for Task 4 and the number generator to Russell for GUI wiring.
