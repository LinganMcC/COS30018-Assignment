# Handwritten expression test – `tuning/ext_v3_nostyle, sheets 3,4`

408 cropped expressions from the team's sheets (`data/custom_samples/expressions/ready/`).

| Set | Images | Exact expression | Symbol accuracy | Right length | Result correct |
|---|---|---|---|---|---|
| **All** | 408 | 86.5 % | 97.3 % | 89.0 % | 86.5 % |
| Level 1 – one operator | 108 | 95.4 % | 98.6 % | 96.3 % | 95.4 % |
| Level 2 – two operators | 93 | 91.4 % | 98.4 % | 92.5 % | 91.4 % |
| Level 3 – brackets | 111 | 83.8 % | 97.3 % | 88.3 % | 83.8 % |
| Level 4 – nested brackets | 96 | 75.0 % | 96.3 % | 78.1 % | 75.0 % |

Right length = the number of symbols found equals the truth (mostly a segmentation measure). Symbol accuracy = 1 − edit distance / true length.

## Per symbol (expressions with the right length)

| Symbol | Seen | Accuracy |
|---|---|---|
| 0 | 135 | 100.0 % |
| 1 | 174 | 100.0 % |
| 2 | 152 | 100.0 % |
| 3 | 109 | 100.0 % |
| 4 | 184 | 100.0 % |
| 5 | 147 | 100.0 % |
| 6 | 191 | 100.0 % |
| 7 | 118 | 99.2 % |
| 8 | 141 | 100.0 % |
| 9 | 117 | 100.0 % |
| ( | 292 | 99.0 % |
| ) | 292 | 99.7 % |
| × | 200 | 100.0 % |
| + | 223 | 100.0 % |
| − | 186 | 100.0 % |
| ÷ | 155 | 96.8 % |

## Most frequent confusions

| True | Read as | Count |
|---|---|---|
| ÷ | − | 4 |
| ( | 1 | 2 |
| ÷ | + | 1 |
| 7 | + | 1 |
| ( | 0 | 1 |
| ) | − | 1 |

## Per writer

| Writer | Images | Exact | Symbol accuracy | Right length |
|---|---|---|---|---|
| s3w1 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w2 | 30 | 93.3 % | 99.2 % | 96.7 % |
| s3w3 | 30 | 43.3 % | 82.3 % | 50.0 % |
| s3w4 | 28 | 78.6 % | 96.5 % | 85.7 % |
| s3w5 | 30 | 90.0 % | 98.7 % | 93.3 % |
| s3w6 | 30 | 96.7 % | 99.6 % | 96.7 % |
| s3w7 | 30 | 100.0 % | 100.0 % | 100.0 % |
| s4w1 | 30 | 93.3 % | 99.2 % | 96.7 % |
| s4w2 | 28 | 64.3 % | 93.8 % | 67.9 % |
| s4w3 | 29 | 100.0 % | 100.0 % | 100.0 % |
| s4w4 | 25 | 72.0 % | 96.3 % | 72.0 % |
| s4w5 | 29 | 93.1 % | 99.1 % | 96.6 % |
| s4w6 | 29 | 89.7 % | 97.4 % | 93.1 % |
| s4w7 | 30 | 96.7 % | 99.6 % | 96.7 % |
