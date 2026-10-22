# Handwritten expression test – `ext_v2` without the input blur, all 823 photos (ablation)

823 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 823 | 43.4 % | 75.6 % | 91.1 % | 43.4 % |
| Level 1 – one operator | 219 | 89.5 % | 97.2 % | 97.3 % | 89.5 % |
| Level 2 – two operators | 190 | 84.7 % | 96.7 % | 94.7 % | 84.7 % |
| Level 3 – brackets | 221 | 0.0 % | 70.5 % | 91.0 % | 0.0 % |
| Level 4 – nested brackets | 193 | 0.0 % | 62.8 % | 80.8 % | 0.0 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 265 | 100.0 % |
| 1 | 341 | 74.8 % |
| 2 | 354 | 100.0 % |
| 3 | 283 | 100.0 % |
| 4 | 366 | 99.5 % |
| 5 | 322 | 100.0 % |
| 6 | 348 | 100.0 % |
| 7 | 226 | 100.0 % |
| 8 | 264 | 100.0 % |
| 9 | 212 | 100.0 % |
| ( | 599 | 0.5 % |
| ) | 599 | 8.5 % |
| × | 411 | 100.0 % |
| + | 462 | 98.7 % |
| − | 384 | 99.5 % |
| ÷ | 321 | 93.8 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| ( | 1 | 576 |
| ) | 1 | 377 |
| ) | 7 | 169 |
| 1 | 4 | 84 |
| ( | 6 | 20 |
| ÷ | − | 18 |
| + | 4 | 4 |
| + | × | 2 |
| ÷ | + | 1 |
| 4 | + | 1 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
| s1w1 | 30 | 46.7 % | 76.4 % | 93.3 % |
| s1w2 | 30 | 33.3 % | 76.0 % | 96.7 % |
| s1w3 | 30 | 43.3 % | 77.3 % | 83.3 % |
| s1w4 | 29 | 48.3 % | 76.0 % | 96.6 % |
| s1w5 | 29 | 44.8 % | 75.0 % | 96.6 % |
| s1w6 | 30 | 33.3 % | 72.1 % | 96.7 % |
| s1w7 | 30 | 46.7 % | 76.0 % | 100.0 % |
| s2w1 | 29 | 44.8 % | 77.5 % | 100.0 % |
| s2w2 | 30 | 50.0 % | 77.0 % | 100.0 % |
| s2w3 | 30 | 43.3 % | 77.8 % | 90.0 % |
| s2w4 | 29 | 51.7 % | 76.8 % | 96.6 % |
| s2w5 | 29 | 41.4 % | 72.3 % | 79.3 % |
| s2w6 | 30 | 50.0 % | 74.9 % | 83.3 % |
| s2w7 | 30 | 46.7 % | 77.8 % | 93.3 % |
| s3w1 | 30 | 43.3 % | 78.0 % | 96.7 % |
| s3w2 | 30 | 46.7 % | 78.0 % | 96.7 % |
| s3w3 | 30 | 33.3 % | 64.6 % | 50.0 % |
| s3w4 | 28 | 42.9 % | 76.3 % | 85.7 % |
| s3w5 | 30 | 46.7 % | 76.4 % | 93.3 % |
| s3w6 | 30 | 36.7 % | 73.4 % | 96.7 % |
| s3w7 | 30 | 46.7 % | 77.6 % | 100.0 % |
| s4w1 | 30 | 46.7 % | 78.1 % | 96.7 % |
| s4w2 | 28 | 39.3 % | 69.8 % | 67.9 % |
| s4w3 | 29 | 41.4 % | 75.6 % | 100.0 % |
| s4w4 | 25 | 28.0 % | 74.1 % | 72.0 % |
| s4w5 | 29 | 41.4 % | 77.0 % | 96.6 % |
| s4w6 | 29 | 48.3 % | 78.3 % | 93.1 % |
| s4w7 | 30 | 46.7 % | 77.3 % | 96.7 % |
