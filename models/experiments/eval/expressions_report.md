# Handwritten expression test – `ext_best, sheets 3,4`

408 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 408 | 87.5 % | 97.4 % | 89.0 % | 87.5 % |
| Level 1 – one operator | 108 | 95.4 % | 98.6 % | 96.3 % | 95.4 % |
| Level 2 – two operators | 93 | 91.4 % | 98.4 % | 92.5 % | 91.4 % |
| Level 3 – brackets | 111 | 87.4 % | 97.6 % | 88.3 % | 87.4 % |
| Level 4 – nested brackets | 96 | 75.0 % | 96.3 % | 78.1 % | 75.0 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 135 | 100.0 % |
| 1 | 174 | 99.4 % |
| 2 | 152 | 100.0 % |
| 3 | 109 | 100.0 % |
| 4 | 184 | 100.0 % |
| 5 | 147 | 100.0 % |
| 6 | 191 | 100.0 % |
| 7 | 118 | 100.0 % |
| 8 | 141 | 100.0 % |
| 9 | 117 | 100.0 % |
| ( | 292 | 100.0 % |
| ) | 292 | 99.7 % |
| × | 200 | 100.0 % |
| + | 223 | 100.0 % |
| − | 186 | 100.0 % |
| ÷ | 155 | 97.4 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| ÷ | − | 3 |
| ÷ | + | 1 |
| 1 | ( | 1 |
| ) | − | 1 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
| s3w1 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w2 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w3 | 30 | 50.0 % | 83.1 % | 50.0 % |
| s3w4 | 28 | 82.1 % | 96.9 % | 85.7 % |
| s3w5 | 30 | 90.0 % | 98.7 % | 93.3 % |
| s3w6 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w7 | 30 | 100.0 % | 100.0 % | 100.0 % |
| s4w1 | 30 | 93.3 % | 99.2 % | 96.7 % |
| s4w2 | 28 | 64.3 % | 93.8 % | 67.9 % |
| s4w3 | 29 | 100.0 % | 100.0 % | 100.0 % |
| s4w4 | 25 | 72.0 % | 95.4 % | 72.0 % |
| s4w5 | 29 | 93.1 % | 99.1 % | 96.6 % |
| s4w6 | 29 | 89.7 % | 97.9 % | 93.1 % |
| s4w7 | 30 | 96.7 % | 99.6 % | 96.7 % |
