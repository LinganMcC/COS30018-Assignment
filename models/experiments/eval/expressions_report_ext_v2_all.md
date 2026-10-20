# Handwritten expression test – `ext_v2` (previous ext_best), all 823 photos, before retraining

823 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 823 | 72.5 % | 95.0 % | 91.1 % | 72.5 % |
| Level 1 – one operator | 219 | 82.2 % | 94.7 % | 97.3 % | 82.2 % |
| Level 2 – two operators | 190 | 75.8 % | 94.8 % | 94.7 % | 75.8 % |
| Level 3 – brackets | 221 | 67.9 % | 94.9 % | 91.0 % | 67.9 % |
| Level 4 – nested brackets | 193 | 63.7 % | 95.2 % | 80.8 % | 63.7 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 265 | 100.0 % |
| 1 | 341 | 67.7 % |
| 2 | 354 | 99.4 % |
| 3 | 283 | 99.6 % |
| 4 | 366 | 98.9 % |
| 5 | 322 | 100.0 % |
| 6 | 348 | 98.9 % |
| 7 | 226 | 86.7 % |
| 8 | 264 | 100.0 % |
| 9 | 212 | 100.0 % |
| ( | 599 | 99.8 % |
| ) | 599 | 99.3 % |
| × | 411 | 99.8 % |
| + | 462 | 98.9 % |
| − | 384 | 99.7 % |
| ÷ | 321 | 94.1 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| 1 | 4 | 40 |
| 7 | + | 30 |
| 1 | + | 29 |
| 1 | × | 27 |
| ÷ | − | 17 |
| 1 | ( | 9 |
| + | 4 | 5 |
| 6 | + | 4 |
| 1 | 7 | 3 |
| ) | 1 | 3 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
| s1w1 | 30 | 83.3 % | 97.4 % | 93.3 % |
| s1w2 | 30 | 50.0 % | 92.3 % | 96.7 % |
| s1w3 | 30 | 66.7 % | 94.0 % | 83.3 % |
| s1w4 | 29 | 79.3 % | 97.2 % | 96.6 % |
| s1w5 | 29 | 79.3 % | 97.4 % | 96.6 % |
| s1w6 | 30 | 73.3 % | 95.3 % | 96.7 % |
| s1w7 | 30 | 93.3 % | 99.1 % | 100.0 % |
| s2w1 | 29 | 82.8 % | 97.8 % | 100.0 % |
| s2w2 | 30 | 50.0 % | 92.5 % | 100.0 % |
| s2w3 | 30 | 76.7 % | 95.8 % | 90.0 % |
| s2w4 | 29 | 75.9 % | 96.5 % | 96.6 % |
| s2w5 | 29 | 75.9 % | 94.5 % | 79.3 % |
| s2w6 | 30 | 63.3 % | 92.9 % | 83.3 % |
| s2w7 | 30 | 90.0 % | 98.7 % | 93.3 % |
| s3w1 | 30 | 93.3 % | 98.8 % | 96.7 % |
| s3w2 | 30 | 43.3 % | 90.5 % | 96.7 % |
| s3w3 | 30 | 43.3 % | 78.5 % | 50.0 % |
| s3w4 | 28 | 64.3 % | 93.9 % | 85.7 % |
| s3w5 | 30 | 90.0 % | 98.7 % | 93.3 % |
| s3w6 | 30 | 60.0 % | 94.2 % | 96.7 % |
| s3w7 | 30 | 96.7 % | 99.6 % | 100.0 % |
| s4w1 | 30 | 93.3 % | 99.2 % | 96.7 % |
| s4w2 | 28 | 50.0 % | 89.3 % | 67.9 % |
| s4w3 | 29 | 51.7 % | 92.0 % | 100.0 % |
| s4w4 | 25 | 68.0 % | 95.4 % | 72.0 % |
| s4w5 | 29 | 72.4 % | 95.6 % | 96.6 % |
| s4w6 | 29 | 72.4 % | 94.0 % | 93.1 % |
| s4w7 | 30 | 90.0 % | 98.3 % | 96.7 % |
