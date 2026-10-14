# Final models on real handwriting

70 single-digit photos, 27 number photos (`data/custom_samples/`). Segmentation: contours; preprocessing: otsu_mnist_box.

| Model | MNIST test | Real single digits (95 % CI) | Exact numbers | Digit count right | Digit acc. when count right |
|---|---|---|---|---|---|
| MLP (tuned) | 98.99 % | 74.3 % (63–83) | 66.7 % | 74.1 % | 91.8 % |
| LeNet-5 (tuned) | 99.29 % | 87.1 % (77–93) | 63.0 % | 74.1 % | 90.2 % |
| ResNet (tuned) | 99.57 % | 85.7 % (76–92) | 63.0 % | 74.1 % | 88.5 % |
| VGG-deep = cnn_best | 99.62 % | 90.0 % (81–95) | 66.7 % | 74.1 % | 91.8 % |
| ext_best, digits only | 99.57 % | 85.7 % (76–92) | 63.0 % | 74.1 % | 90.2 % |
| ext_best, all 16 classes | 99.57 % | 84.3 % (74–91) | 59.3 % | 74.1 % | 88.5 % |
| Sprint 2 LeNet (cnn_lenet) | – | 84.3 % (74–91) | 63.0 % | 74.1 % | 90.2 % |

Digit count right is set by segmentation alone, so it is the same for every model; the last column isolates recognition errors.
