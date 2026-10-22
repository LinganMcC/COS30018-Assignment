# Overall system functions and performance (Task 4)

*Report section. Numbers come from `evaluation/results/model_comparison.md` and
`failure_analysis.md` (`python evaluation/evaluate_all.py`,
`python evaluation/error_analysis.py`) and, for the extension, from
`models/experiments/eval/expressions_report.md`.*

## 1. What the system does

| Function | How | Where |
|---|---|---|
| Read a handwritten number from a photo | segmentation → preprocessing → CNN, digits joined left to right | `models/cnn/inference.py` (`Recogniser`) |
| Build the input number automatically from a folder of single-digit images | GUI *Input → Generate from a folder of digit images*: type a number or expression (or leave it empty for a random one); each digit image – MNIST or a photo – is cropped to its ink, cleaned, scaled to one height and stroke width and placed side by side | `src/number_from_folder.py` (test sets: `src/generate_number.py`) |
| Load the input from an image file | *Input → Load image…* or drag and drop | `gui.py` |
| Show every step | the binary image, every segmented box, the 28×28 input of each digit, and a confidence for each digit | GUI tabs *Preprocessed*, *Segments*; *Output* panel |
| Choose the model and its source | *CNN*, *CNN + operators*, *MLP*, *SVM* or *Random Forest*; frozen pretrained model, or train a new one with chosen architecture, augmentation, learning rate, batch size and epochs, with live curves and *Stop* | GUI *Settings* |
| Choose the segmentation | *Repaired* (the system's) or *Contours* (Task 2 selection, for comparison) | GUI *Settings → Segment* |
| Read and calculate handwritten arithmetic (extension) | the same pipeline with operator-aware segmentation and a 16-class CNN; the result is calculated by a precedence- and bracket-aware parser | GUI *CNN + operators*; `expression_parser.py`; `evaluate_expressions.py` |

Pipeline of the final system: **repaired** segmentation – ruled lines removed,
`contours`, stacked pieces merged, boxes too wide for one digit cut where
`cnn_best` reads the pieces best (`src/segmentation.py`, `src/segmentation_split.py`)
→ **otsu_mnist_box** preprocessing (`src/preprocessing.py`) → **VGG-deep**
(`cnn_best`) for numbers; for expressions, operator-aware `contours` → `ext_best`.

## 2. How it was evaluated

| Test set | Size | What it tests |
|---|---|---|
| MNIST test | 10,000 single digits | recognition under the training distribution |
| Real single digits | 149 photos, 4 writers | recognition of real handwriting |
| Real numbers | 99 photos of 2–5 digits, 4 writers, 6 sheets; 25 with touching digits | the whole pipeline on real input |
| Generated numbers | 600 numbers of 2–5 MNIST **test** digits; 200 each at wide (14 px), tight (−6 px) and touching (−14 px) spacing | the whole pipeline under controlled spacing, with digits no model was trained on |
| Expressions (extension) | 408 photos by 14 writers never used in training | the extension end to end |

As the specification requires, two measures are reported for numbers and kept separate:

- **Per-digit accuracy** – 1 − edit distance ÷ number of true digits, so a wrong,
  missing or extra digit each counts as one error. Also given only on numbers cut
  into the right number of pieces, which isolates recognition.
- **Whole-number accuracy** – a number counts only if every digit is right.

## 3. Results

**Single digits**

| Test set | Accuracy (95 % CI) |
|---|---|
| MNIST test | 99.62 % |
| MNIST test, shifted | 99.47 % |
| Real photographed digits | 86.6 % (80–91) |

**Numbers – the two Task 4 measures**

| Test set | Digit count right (segmentation) | Per-digit accuracy | Per-digit, when the count is right | Whole-number accuracy |
|---|---|---|---|---|
| Generated, wide spacing | 83.5 % | – | – | 71.0 % |
| Generated, tight spacing | 86.0 % | – | – | 74.5 % |
| Generated, touching | 62.5 % | – | – | 51.0 % |
| **Generated, all 600** | 77.3 % | **87.4 %** | 94.6 % | **65.5 %** |
| Real, digits apart (74) | 94.6 % | – | – | 83.8 % |
| Real, digits touching (25) | 28.0 % | – | – | 20.0 % |
| **Real, all 99** | 77.8 % | **77.9 %** | 92.7 % | **67.7 %** (58–76) |

**Extension – handwritten expressions** (408, writers not used in training)

| Measure | Result |
|---|---|
| Symbol recognition accuracy (1 − edit distance ÷ length) | **97.4 %** |
| Expression read exactly | 87.5 % |
| Calculated result correct | **87.5 %** |
| Right number of symbols found | 89.0 % |
| By difficulty: one operator / two operators / brackets / nested brackets | 95.4 % / 91.4 % / 87.4 % / 75.0 % |

The calculated result is the system's own parser, `expression_parser.py`
(precedence and brackets, no `eval()`), the one the GUI uses: it gives the
correct result for 357 of the 408 expressions (87.5 %), the same as the
stand-in evaluator in `evaluate_expressions.py` that produced the table.

## 4. Critical analysis: where and why it fails

Every failed number was put in one category by comparing the segmentation boxes
with the true digits (`evaluation/error_analysis.py`):

| Category | Rule | Real (32 failures) | Generated (207 failures) |
|---|---|---|---|
| merged | fewer boxes than digits, no digit-sized ink discarded | 17 (53 %) | 48 (23 %) |
| dropped | fewer boxes, and the discarded ink adds up to a digit | 0 | 67 (32 %) |
| split | more boxes than digits | 5 (16 %) | 21 (10 %) |
| misread | right number of boxes, a digit wrong | 10 (31 %) | 71 (34 %) |
| **segmentation total** | | **69 %** | **66 %** |

![Examples of each failure category](results/failure_gallery.png)

**1. Segmentation, not recognition, limits the system.** About two thirds of
failures happen before the CNN sees the image. When segmentation is right the
CNN reads 92.7 % of real digits and 94.6 % of generated ones, and 87.0 % of
correctly cut real numbers end to end. The CNN comparison (Task 3) moves
whole-number accuracy by a few points; segmentation decides tens of points.

**2. Touching digits are still the dominant failure, though the repair halves
the gap.** `contours` finds one box per connected region of ink, so digits whose
strokes touch become one box. The second-round repair cuts a box too wide for
one digit where `cnn_best` reads the pieces most confidently. It raised touching
real numbers from 8 % to 20 % read exactly and touching generated numbers from
29.5 % to 51.0 %; overall 63.6 % → 67.7 % on real numbers (4 fixed, none lost)
and 60.0 % → 65.5 % on generated ones (44 fixed, 11 lost). Still, 17 of the 32
real failures are touching digits left as one box: they are read correctly
20 % of the time, against 83.8 % when the digits are apart.

The repair has a cost: it assumes a very wide box holds several digits, so a
sweeping single digit is sometimes cut in two. "Split" failures rose from 3 to 5
on real numbers and from 0 to 21 on generated ones, and on the 149 single-digit
photos it cuts 25 into two or more pieces against 13 for `contours`. On
numbers it fixes more than it breaks, so it is the system's default; the GUI
keeps *Contours* as an option.

**3. Faint digits are dropped – on generated input only.** At wide spacing,
where digits cannot touch, 27 of 58 failures are a digit vanishing entirely. The
thin, faint MNIST strokes break into several fragments after thresholding, and
each fragment is smaller than the noise filter's minimum size, so all are
discarded (example: "28379" read as "2879"). No real photo failed this way –
pen strokes stay continuous – so this mainly affects the generated input mode.

**4. Recognition errors come from writing style.** The 10 real misreads include
1 → 6 twice and eight different single confusions; on generated numbers 4 → 1 (9),
9 → 4 (8), 0 → 6 (6), 3 → 5 and 7 → 1 (5 each). These are shapes where the writer's style overlaps another digit's
MNIST shape. The extension showed the remedy: fine-tuning on the team's own
handwriting raised "1" from 68 % to 99 % and "7" from 86 % to 100 % on unseen
writers (`models/experiments/eval/retraining_report.md`).

**5. Writing conditions matter more than the writer.** Whole-number accuracy by
sheet ranges from 83.3 % to 38.9 %; the 38.9 % sheet is the one written on
hand-ruled lines with slanted writing. Otsu thresholding keeps a ruled line as
ink, which joins digits and adds bars to them; removing the lines before
segmentation (`src/page_cleanup.py`) raised that sheet from 33.3 % to 38.9 %,
and the remaining errors there are slanted, touching digits.

**6. The gap between MNIST and real handwriting is large.** The same model reads
99.6 % of MNIST digits and 86.6 % of photographed ones. MNIST accuracy alone would
have overstated the system badly; every selection in this project was therefore
checked on real photos.

**7. The model knows when it is unsure.** On real numbers cut correctly, digits
read right have a median confidence of 1.00; digits read wrong, 0.81. Flagging
digits below 0.9 confidence would mark 29 of 232 digits and catch 71 % of the
wrong ones (`results/confidence.png`) – a cheap way for the GUI to tell the user
which digit to check.

**8. The extension inherits the same limit.** Of the 51 failed expressions, 45 are
a wrong number of symbols (segmentation) and only 6 a misread symbol. Nested
brackets are hardest: 21.9 % of them are cut wrongly, because a bracket written
close to a digit touches it.

## 5. What would improve the system most

In order of expected gain, from the evidence above:

Already done in this project: touching digits are cut by the repaired
segmentation, and ruled lines are removed. What is left:

1. **Better cuts for touching digits** – still 53 % of real failures. For
   example, more candidate cut paths, or a small classifier trained to tell
   "one digit" from "two touching digits", which would also stop single wide
   digits being cut in two.
2. **Group stroke fragments before the noise filter** so faint digits are not
   discarded – targets 32 % of generated-number failures.
3. **Fine-tune the digit model on the team's handwriting**, as was done for the
   extension (which raised "1" from 68 % to 99 % on unseen writers) – targets
   the 31 % of real failures that are misreads. It needs writers kept out of
   training to be measured honestly.
4. **Show low-confidence digits in the GUI** so the user can correct them.

## 6. Limitations of this evaluation

- The real test sets are small: 149 digits and 99 numbers from four writers, so
  real-photo accuracies carry 95 % intervals of about ±6–9 points.
- All real handwriting comes from the team; other people's writing may differ.
- The repaired segmentation's settings were chosen on two of the six real sheets
  and the generated development set (`experiments/evaluate_split.py`); the
  real-number figures include those two sheets. On the four held-out sheets the
  gain is the same size (62.5 % → 66.7 % read exactly).
- Generated numbers use MNIST test digits: they test segmentation under
  controlled spacing, not the variety of real pens and paper.
- The extension's 408 test expressions repeat the same 120 expressions written
  by new people: they measure new writers, not new kinds of expression.
