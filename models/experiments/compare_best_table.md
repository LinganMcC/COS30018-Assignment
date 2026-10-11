| Model | Run | Aug | Dropout | Batch | Epochs | Val acc | Test acc | Shifted-test acc | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| MLP (simple) | pm_mlp_best | light | 0.3 | 128 | 31 | 98.98% | 98.99% | 96.89% | 107 |
| LeNet-5 | pm_lenet_best | light | 0.3 | 128 | 30 | 99.17% | 99.29% | 98.09% | 108 |
| ResNet (small) | pm_resnet_best | light | 0.3 | 128 | 31 | 99.60% | 99.57% | 99.54% | 1895 |
| VGG (deep) | s3_B5_bs_64 | light | 0.5 | 64 | 19 | 99.63% | 99.62% | 99.47% | 760 |
