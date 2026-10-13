# Image Segmentation (Task 2)

*Owner: John. Report section for "System descriptions → Image segmentation".*

## 1. What this stage has to do

Segmentation takes one image of a multi-digit number and returns one sub-image
per digit, ordered left to right, so that the classifier — which only ever
learned single digits — can be applied to each in turn. Its output is a list of
bounding boxes and a list of square crops.

Two properties matter more than they first appear:

- **The count must be right.** A number read as four digits when it has five is
  wrong no matter how well each piece is classified. Digit-count accuracy is
  therefore reported separately from recognition accuracy throughout.
- **The order must be right.** Boxes are sorted by their left edge, which is
  correct for horizontally written numbers and is the assumption the whole
  pipeline rests on.

The implementations are `src/segmentation.py` (classical) and
`src/segmentation_ml.py` (model-based).

## 2. Techniques investigated

Ten methods were implemented and compared, in two families.

### Classical (`src/segmentation.py`)

| Method | Principle | Expected weakness |
|---|---|---|
| `contours` | Outer contour of each connected ink region, via `cv2.findContours` | Two touching digits are one region |
| `connected_components` | Label connected white regions, via `cv2.connectedComponentsWithStats` | Same — it finds the same thing a different way |
| `projection` | Count ink per column; cut at the valleys | Assumes digits do not overlap horizontally; a thin stroke reads as a valley |
| `watershed` | Distance transform peaks as seeds, grown back out by watershed | Over-splits a single complex digit |

`watershed` was added specifically to attack the failure the first two share.
Where contours and connected components see two touching digits as one shape,
the distance transform peaks near the centre of *each* digit, so thresholding
it gives one seed per digit even when the ink is joined. Each crop is masked to
its own label so a neighbour's ink cannot leak in.

### Model-based (`src/segmentation_ml.py`)

These ask a trained classifier where the digits are, rather than deducing it
from the pixels.

| Method | Principle |
|---|---|
| `sliding_mlp`, `sliding_cnn`, `sliding_svm`, `sliding_rf` | Slide a window across the ink and keep the positions one model is confident about; four models, one search strategy, so the comparison isolates the model |
| `cutpoint_mlp` | Over-segment at every plausible boundary, then keep the subset of cuts the model scores best |
| `confidence_split` | Start from connected blobs and split only the blobs the model thinks hold more than one digit |

`confidence_split` is the interesting hybrid: it keeps the classical blob
detection that works and uses the model only for the case where it fails.

## 3. How they were compared

**(a) 30 generated numbers.** Built by `src/generate_number.py` from real MNIST
digit images, with the ground-truth label recorded. Spacing is deliberately
varied across three difficulty levels — `wide` (+14 px), `tight` (−6 px) and
`touching` (−14 px) — ten numbers each. A test set on which every method scores
100 % would not help choose between them; the touching band is there to make the
methods disagree.
Script: `experiments/demo_segmentation.py` → `reports/segmentation_comparison.csv`.

**(b) 27 real photographed numbers**, 8 of them containing digits that touch.
Both digit-count accuracy and end-to-end exact-number accuracy are reported.
Script: `experiments/evaluate_custom.py` → `reports/custom_segmentation.csv`.

## 4. Results

### (a) 30 generated numbers — correct digit count

| Method | Overall | wide | tight | touching |
|---|---|---|---|---|
| `projection` | **83 %** | 90 % | 100 % | 60 % |
| `confidence_split` | 80 % | 90 % | 90 % | 60 % |
| `contours` | 70 % | 90 % | 80 % | 40 % |
| `connected_components` | 70 % | 90 % | 80 % | 40 % |
| `sliding_cnn` | 67 % | 30 % | 90 % | 80 % |
| `cutpoint_mlp` | 60 % | 40 % | 80 % | 60 % |
| `sliding_mlp` | 57 % | 60 % | 70 % | 40 % |
| `watershed` | 50 % | 50 % | 60 % | 40 % |
| `sliding_svm` | 37 % | 40 % | 60 % | 10 % |
| `sliding_rf` | 23 % | 40 % | 20 % | 10 % |

### (b) 27 real photographed numbers

| Method | Correct digit count | Exact number read |
|---|---|---|
| **`contours`** | **74 %** | **63 %** |
| `connected_components` | 74 % | 63 % |
| `confidence_split` | 74 % | 63 % |
| `watershed` | 52 % | 37 % |
| `sliding_mlp` | 37 % | 7 % |
| `cutpoint_mlp` | 33 % | 19 % |
| `sliding_cnn` | 30 % | 7 % |
| `projection` | 19 % | 7 % |
| `sliding_svm` | 19 % | 7 % |
| `sliding_rf` | 19 % | 4 % |

Figures: `reports/segmentation_methods.png`, `reports/segmentation_touching.png`,
`reports/segmentation_models_chart.png`, `reports/segmentation_real_visual.png`.

### The most instructive result: `projection`

`projection` wins table (a) at 83 % and collapses to 19 % — last place — in
table (b). It was in fact the method selected on the generated set, and the
selection had to be reversed.

The cause is specific and worth stating precisely. The method cuts the image
wherever a column holds less ink than `valley_ratio × (busiest column)`, and
that threshold was tuned to 0.20 by sweeping it against the generated set. But
MNIST strokes are thick and anti-aliased, while a real ballpoint stroke is two
or three pixels wide. A column passing through the middle of a real `0` or `6`
crosses only the two thin sides of the loop, so it falls under the threshold
and reads as a gap between digits. One photographed `60` was returned as eleven
pieces.

The general lesson, which applies to every method in these tables: a parameter
tuned on generated data is tuned to the generator, and a technique that wins on
MNIST-derived images has not yet been shown to work on photographs. This is the
single strongest argument for having built the real test set at all.

## 5. Selected technique

**`contours`** — Otsu binarization, external contours, filter out boxes too
small to be digits, sort by x, pad each crop to a square.

`contours` and `connected_components` tie exactly on both test sets, which is
expected: they find the same thing — connected regions of ink — computed two
ways. `contours` is taken as the selection; `connected_components` remains in
the codebase as the cross-check that the tie is real.

`confidence_split` also ties on real photographs (74 % / 63 %) and is better on
generated images (80 % vs 70 %). It was not selected because it requires a
trained model to be loaded before any segmentation can happen, which couples the
front of the pipeline to the back and makes the GUI and the evaluation harness
depend on a checkpoint being present. It did not buy anything on real data to
pay for that, though it is the natural first upgrade if touching digits become
the priority.

The crops are padded to a square before resizing. Without this a `1` — tall and
narrow — is stretched sideways into something closer to an `8` when it reaches
28×28.

The choice is recorded as `SELECTED_METHOD` in `src/segmentation.py` and used as
the default argument of `segment_digits()`, with a unit test asserting the two
agree.

## 6. Extension: segmenting arithmetic operators

The arithmetic extension needs the same stage to return `+ − * ÷ ( )` as well as
digits. It did not, and the reason was in the noise filter rather than in any of
the four methods.

Boxes were rejected below 15 % of the image height, a rule that works because a
digit always fills a good share of the frame. An operator does not. Measured on
a test expression 160 px tall: a minus sign is 5 % of the image height, an
asterisk 13 %, and the bar and dots of a division sign less again. Three of the
six operator classes were being discarded before any classifier saw them, and
a division sign — three separate regions of ink — was discarded three times
over.

Two changes fix it, both in `src/segmentation.py`:

1. **Measure operators against the digits, not against the image.** The tallest
   box in an image is a digit, and that sets the scale whatever the photo's
   margins. A box is admitted as an operator if it is *bar-like* (several times
   wider than tall, which no digit is) or *blob-like* (roughly square, clearly
   smaller than a digit but far larger than dust).
2. **Rejoin stacked pieces.** Boxes whose columns overlap are merged, which
   turns the bar and two dots of a division sign back into one symbol. In a
   left-to-right expression no two different symbols share a column, so
   horizontal overlap alone is a safe signal.

Both are behind `keep_operators`, exposed as `segment_expression()`. The
entry point for digits is unchanged.

### Measured on 40 generated expressions

Built by `src/generate_number.py --expressions` from real MNIST digits plus
drawn operator glyphs, across five expression shapes from no brackets to nested
brackets. Score is the share of images split into exactly the right number of
symbols.
Script: `experiments/evaluate_operators.py`.

| Method | Operators off | Operators on |
|---|---|---|
| **`contours`** | 28 % | **85 %** |
| `connected_components` | 28 % | 85 % |
| `watershed` | 3 % | 58 % |
| `projection` | 40 % | 35 % |

`projection` cannot be used here and `segment_expression()` rejects it by name
rather than returning a quietly wrong answer: it cuts on vertical gaps, and a
division sign's dots occupy the same columns as its bar, so the three pieces
arrive as a single band indistinguishable from a minus sign.

### Why this is opt-in, not always on

Admitting shapes much shorter than a digit also admits dirt on the paper.
Measured on the same 27 real photographed numbers:

| Method | Operators off | Operators on |
|---|---|---|
| `contours` | 74 % | 70 % |
| `connected_components` | 74 % | 70 % |
| `watershed` | 52 % | 37 % |
| `projection` | 19 % | 19 % |

`contours` loses one number of 27 to one extra box. `watershed` loses far more,
because it already breaks ink into fragments and the relaxed filter then accepts
those fragments as operators. Expression input turns the flag on and pays that
cost; number input does not.

An earlier version of the merge step was applied to all boxes rather than only
to operator-shaped ones, and cost two of the 27 numbers: slanted handwriting
puts one digit far enough over its neighbour's column to look like a stacked
pair. Restricting the merge to operator-shaped boxes removed that regression,
and it also means that turning the flag on can only *add* boxes — a digit found
with the flag off is still found, as its own box, with it on.

## 7. Known limitations

**Touching digits are the main open limitation.** On the generated set
`contours` scores 90 % on widely spaced numbers, 80 % on tight ones and 40 % on
touching ones; on the real photographs only 2 of the 8 numbers with touching
digits are counted correctly. Two touching digits are one connected region, so a
connectivity-based method cannot see them as two by construction. Both
alternatives were implemented — `watershed` and `confidence_split` — and neither
beat `contours` overall on real data, so the limitation stands rather than
being traded for a worse average.

**Segmentation, not classification, is what loses most numbers.** Of the ten
misread numbers in the end-to-end test, seven were split into the wrong number
of pieces, six of those because digits touch. Effort spent on touching digits
would be worth more than effort spent on the classifier.

**The expression evidence is partly synthetic.** The digits in the 40 test
expressions are real MNIST samples, but the operator glyphs are drawn, because
the team's handwritten operator samples were not yet collected when this was
measured. Shape and relative size are right, which is what segmentation depends
on, but the measurement should be repeated on real handwritten operators before
the extension's figures are quoted as final.

**Four of the six remaining expression failures are over-segmentation**, where a
drawn `+` or `*` breaks into two pieces under Otsu at a thin stroke width and
the merge step — which only rejoins pieces stacked *vertically*, as a division
sign needs — leaves them apart. Real handwritten operators are drawn in
continuous strokes and are less likely to show this, which is another reason to
re-measure on real samples.

**Horizontal writing is assumed throughout.** Boxes are ordered by left edge, so
a number written on a slope, wrapped onto two lines, or rotated more than
slightly will be read in the wrong order.
