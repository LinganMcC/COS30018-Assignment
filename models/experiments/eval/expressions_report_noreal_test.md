# Handwritten expression test – `tuning/ext_v3_noreal, sheets 3,4`

408 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 408 | 80.9 % | 96.3 % | 89.0 % | 80.9 % |
| Level 1 – one operator | 108 | 90.7 % | 97.2 % | 96.3 % | 90.7 % |
| Level 2 – two operators | 93 | 86.0 % | 97.0 % | 92.5 % | 86.0 % |
| Level 3 – brackets | 111 | 78.4 % | 96.5 % | 88.3 % | 78.4 % |
| Level 4 – nested brackets | 96 | 67.7 % | 95.4 % | 78.1 % | 67.7 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 135 | 100.0 % |
| 1 | 174 | 88.5 % |
| 2 | 152 | 100.0 % |
| 3 | 109 | 99.1 % |
| 4 | 184 | 100.0 % |
| 5 | 147 | 100.0 % |
| 6 | 191 | 100.0 % |
| 7 | 118 | 96.6 % |
| 8 | 141 | 100.0 % |
| 9 | 117 | 100.0 % |
| ( | 292 | 100.0 % |
| ) | 292 | 99.7 % |
| × | 200 | 100.0 % |
| + | 223 | 100.0 % |
| − | 186 | 100.0 % |
| ÷ | 155 | 92.9 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| 1 | + | 12 |
| ÷ | − | 10 |
| 1 | × | 6 |
| 7 | + | 4 |
| 1 | ) | 2 |
| ÷ | + | 1 |
| 3 | 8 | 1 |
| ) | − | 1 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
| s3w1 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w2 | 30 | 60.0 % | 94.2 % | 96.7 % |
| s3w3 | 30 | 50.0 % | 81.9 % | 50.0 % |
| s3w4 | 28 | 67.9 % | 94.7 % | 85.7 % |
| s3w5 | 30 | 90.0 % | 98.7 % | 93.3 % |
| s3w6 | 30 | 86.7 % | 98.3 % | 96.7 % |
| s3w7 | 30 | 96.7 % | 99.6 % | 100.0 % |
| s4w1 | 30 | 93.3 % | 99.2 % | 96.7 % |
| s4w2 | 28 | 64.3 % | 93.8 % | 67.9 % |
| s4w3 | 29 | 75.9 % | 96.2 % | 100.0 % |
| s4w4 | 25 | 72.0 % | 94.9 % | 72.0 % |
| s4w5 | 29 | 93.1 % | 99.1 % | 96.6 % |
| s4w6 | 29 | 89.7 % | 97.9 % | 93.1 % |
| s4w7 | 30 | 93.3 % | 99.2 % | 96.7 % |
