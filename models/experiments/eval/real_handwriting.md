# Final models on real handwriting

149 single-digit photos, 99 number photos (`data/custom_samples/`). Segmentation: contours; preprocessing: otsu_mnist_box.

| Model | MNIST test | Real single digits (95 % CI) | Exact numbers | Digit count right | Digit acc. when count right |
|---|---|---|---|---|---|
| MLP (tuned) | 98.99 % | 57.0 % (49–65) | 27.3 % | 74.7 % | 55.9 % |
| LeNet-5 (tuned) | 99.29 % | 83.9 % (77–89) | 60.6 % | 74.7 % | 88.6 % |
| ResNet (tuned) | 99.57 % | 83.9 % (77–89) | 55.6 % | 74.7 % | 85.9 % |
| VGG-deep = cnn_best | 99.62 % | 86.6 % (80–91) | 63.6 % | 74.7 % | 90.5 % |
| ext_best (v3, retrained), digits only | – | 85.9 % (79–91) | 62.6 % | 74.7 % | 90.5 % |
| ext_best (v3, retrained), all 16 classes | – | 85.2 % (79–90) | 51.5 % | 66.7 % | 87.8 % |
| ext_v2 (before retraining), digits only | 99.57 % | 83.2 % (76–88) | 56.6 % | 74.7 % | 87.7 % |
| ext_v2 (before retraining), all 16 classes | 99.57 % | 66.4 % (59–74) | 34.3 % | 66.7 % | 73.5 % |
| Sprint 2 LeNet (cnn_lenet) | – | 81.9 % (75–87) | 61.6 % | 74.7 % | 90.0 % |

Digit count right is set by segmentation alone, so it is the same for every model; the last column isolates recognition errors.
