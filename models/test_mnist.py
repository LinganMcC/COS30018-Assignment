import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")

# Load MNIST 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"TensorFlow version: {tf.__version__}")
print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
print(f"Test set:     {x_test.shape}, labels: {y_test.shape}")
print(f"Pixel value range: {x_train.min()} to {x_train.max()}")

# Display and save a sample digit
plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.savefig(os.path.join(TESTS_DIR, "sample_digit.png"), bbox_inches="tight")
plt.show()
print("Saved sample_digit.png")