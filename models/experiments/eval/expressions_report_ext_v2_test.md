# Handwritten expression test – `ext_v2, sheets 3,4`

408 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 408 | 70.8 % | 94.2 % | 89.0 % | 70.8 % |
| Level 1 – one operator | 108 | 78.7 % | 93.7 % | 96.3 % | 78.7 % |
| Level 2 – two operators | 93 | 76.3 % | 94.5 % | 92.5 % | 76.3 % |
| Level 3 – brackets | 111 | 64.9 % | 94.1 % | 88.3 % | 64.9 % |
| Level 4 – nested brackets | 96 | 63.5 % | 94.2 % | 78.1 % | 63.5 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 135 | 100.0 % |
| 1 | 174 | 68.4 % |
| 2 | 152 | 98.7 % |
| 3 | 109 | 100.0 % |
| 4 | 184 | 99.5 % |
| 5 | 147 | 100.0 % |
| 6 | 191 | 99.5 % |
| 7 | 118 | 85.6 % |
| 8 | 141 | 100.0 % |
| 9 | 117 | 100.0 % |
| ( | 292 | 99.7 % |
| ) | 292 | 99.3 % |
| × | 200 | 99.5 % |
| + | 223 | 99.6 % |
| − | 186 | 100.0 % |
| ÷ | 155 | 93.5 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| 1 | 4 | 23 |
| 7 | + | 17 |
| 1 | + | 17 |
| 1 | × | 9 |
| ÷ | − | 8 |
| 1 | ( | 5 |
| + | 4 | 1 |
| × | 1 | 1 |
| ÷ | + | 1 |
| 2 | 1 | 1 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
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
