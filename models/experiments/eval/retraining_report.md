# Sprint 4 – retraining the extension model on real handwriting

`ext_best` is now **ext_v3**: the previous model (ext_v2, kept as
`models/checkpoints/ext_v2.keras`) fine-tuned for real handwritten expressions.
Script: `models/cnn/ext_finetune.py`. Data: `models/cnn/real_symbols.py`.

## Why

On the team's photographed expressions ext_v2 had two weaknesses that MNIST and
the symbol datasets cannot fix:

- **Edge style.** Training images have soft edges; photo crops come out of
  `preprocess()` with hard binary edges. ext_v2 used edge softness as a cue and
  read brackets as "1" until an input blur was added (`expressions_ablation.md`).
  The blur then cost 13 points on single-digit photos.
- **Writing style.** The team writes the continental 1 (long flag) and the
  crossed 7. MNIST has almost neither: ext_v2 read "1" correctly 68 % of the time
  and "7" 86 % (held-out writers, with the blur).

## Data and split (by writer, fixed before training)

823 expression photos = 4 sheets × 7 writers. Each photo is segmented with the
GUI pipeline; when the box count equals the truth length, box *i* is labelled
with symbol *i* (73 photos skipped). 5,757 labelled real symbols:

| Split | Writers | Real symbols | Used for |
|---|---|---|---|
| train | sheets 1–2, writers 1–6 (12) | 2,500 | training (repeated ×6 per epoch) |
| val | sheets 1–2, writer 7 (2) | 441 symbols / 60 expressions | choosing the model and the input-blur setting |
| test | sheets 3–4, all writers (14) | 2,816 symbols / 408 expressions | the numbers below, run once |

## Recipe

Fine-tune ext_v2 (VGG-deep, 16 classes) with Adam, lr 3·10⁻⁴ halved on plateau,
batch 64, 8 epochs, keeping the epoch with the best real-validation accuracy.
Training set per epoch (98,453 images):

- MNIST + operator symbols (80,453) – so nothing is forgotten;
- 1,500 synthetic crossed 7s and 1,500 flagged 1s drawn onto MNIST digits;
- the 2,500 real training symbols ×6.

Every batch: geometric augmentation ("light": ±8°, ±8 % shift and zoom) plus
**style augmentation** (`augmentation.style_augment`): stroke width (as is /
thicker / thinner), edge style (as is / hard binary / sharpened / three blur
levels) and contrast (peak 0.7–1.0).

## Results – held-out writers (sheets 3–4, 408 expressions)

| Model | Exact expression | Symbol accuracy | One op | Two ops | Brackets | Nested |
|---|---|---|---|---|---|---|
| ext_v2 (before; with input blur) | 70.8 % | 94.2 % | 78.7 % | 76.3 % | 64.9 % | 63.5 % |
| retrained, style aug, **no real data** | 80.9 % | 96.3 % | 90.7 % | 86.0 % | 78.4 % | 67.7 % |
| retrained, real data, **no style aug** | 86.5 % | 97.3 % | 95.4 % | 91.4 % | 83.8 % | 75.0 % |
| **ext_v3 = new ext_best (full recipe)** | **87.5 %** | **97.4 %** | **95.4 %** | **91.4 %** | **87.4 %** | **75.0 %** |

Right number of symbols found (segmentation): 89.0 % for every model, so 89 %
is the ceiling for exact expressions with the current segmentation. The new
model reaches 87.5 %: nearly all remaining errors are segmentation (mostly
nested brackets, 78 % right length).

Per symbol (expressions with the right length):

| Symbol | ext_v2 | no real data | no style aug | **ext_v3** |
|---|---|---|---|---|
| 1 | 68.4 % | 88.5 % | 100.0 % | **99.4 %** |
| 7 | 85.6 % | 96.6 % | 99.2 % | **100.0 %** |
| ( | 99.7 % | 100.0 % | 99.0 % | **100.0 %** |
| ) | 99.3 % | 99.7 % | 99.7 % | **99.7 %** |
| ÷ | 93.5 % | 92.9 % | 96.8 % | **97.4 %** |
| all other symbols | ≥ 98.7 % | | | **100.0 %** |

Remaining confusions for ext_v3 (whole test set): ÷ → − (3), ÷ → + (1),
1 → ( (1), ) → − (1).

**What each part contributed.** The team's own handwriting is the biggest
single gain (80.9 → 87.5 % exact, and it fixes the 1 and 7). Style augmentation
on its own brings ext_v2 from 70.8 to 80.9 %, and on top of the real data it
adds about one point (86.5 → 87.5 %, four expressions – within noise on 408
images). Its other benefit is that the input blur is no longer needed: on the
validation writers ext_v3 reads 96.7 % of expressions exactly without the blur
and 95.0 % with it, so the checkpoint is saved with `"input_blur": false`.

## No regression elsewhere

| Test | ext_v2 | ext_v3 |
|---|---|---|
| MNIST + symbols test set (16 classes) | 99.57 % | 99.38 % |
| Real single-digit photos, all 16 classes allowed (149) | 66.4 % | **85.2 %** |
| Real single-digit photos, digits only (149) | 83.2 % | **85.9 %** (cnn_best: 86.6 %) |
| Real numbers exact, digits only (99) | 56.6 % | **62.6 %** (cnn_best: 63.6 %) |

## Limitations

- Validation (2 writers) was easy – ext_v2 with the blur already scored 99.3 %
  on its symbols – so it separated the variants only weakly. The test sheets
  were scored once, after the recipe and settings were fixed.
- Labels come from box order, so a photo whose wrong cuts happen to give the
  right count would be mislabelled; a sample of 224 crops checked by eye showed
  one ÷ cut without its dots.
- The digit photos in `data/custom_samples/` may include some of the same
  people as the expression sheets; the expression test is the clean
  writer-disjoint result.
- The test writers are friends of the team writing the same 120 expressions, so
  this measures new writers, not new kinds of expressions.

## Reproduce (from `models/cnn/`)

```bash
python real_symbols.py                                  # build the real-symbol cache
python ext_finetune.py --run-id ext_v3                  # ~30 min on a laptop CPU
python ext_finetune.py --run-id ext_v3_noreal --real-repeat 0
python ext_finetune.py --run-id ext_v3_nostyle --no-style
python evaluate_expressions.py                          # ext_best, held-out sheets 3-4
python evaluate_expressions.py --model ext_v2 --tag _ext_v2_test
python evaluate_real.py                                 # digit and number photos
```
