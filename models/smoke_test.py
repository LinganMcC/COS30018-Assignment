"""
Sprint 1 - trivial training smoke test.

Purpose: confirm the full TensorFlow training loop works end-to-end on MNIST
before investing time in the real CNN in Sprint 2. This is intentionally the
simplest possible model (one hidden dense layer) trained for just 1 epoch -
we only care that it runs, learns something, and saves a model file.

Sprint 1 done-when for Person B: "Environment verified; can run a 1-epoch
toy training run."

Expected outcome: ~92% test accuracy in ~10 seconds on CPU.
"""

import time
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# 1. Load MNIST from the built-in Keras dataset (~11 MB, cached locally after
#    the first run in ~/.keras/datasets/mnist.npz).
# ---------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise pixel values from 0-255 into 0.0-1.0. Neural networks train much
# more stably on small inputs.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"TensorFlow version : {tf.__version__}")
print(f"Training set shape : {x_train.shape}")
print(f"Test set shape     : {x_test.shape}")

# ---------------------------------------------------------------------------
# 2. Build the simplest useful model: flatten 28x28 -> 128-unit dense hidden
#    layer -> 10-way softmax classifier. This is NOT a CNN - Sprint 2 will
#    replace this with a real convolutional network.
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 3. Train for a single epoch. One pass through the training set is enough
#    to prove the pipeline works.
# ---------------------------------------------------------------------------
start = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=2)
train_time = time.time() - start

# ---------------------------------------------------------------------------
# 4. Evaluate on the held-out test set.
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print()
print(f"Training time    : {train_time:.2f} s")
print(f"Test accuracy    : {test_acc * 100:.2f} %")
print(f"Test loss        : {test_loss:.4f}")

# ---------------------------------------------------------------------------
# 5. Save the model to disk to prove serialisation works. Sprint 2 will do
#    the same with the real CNN so Russell's GUI and Liam's evaluation
#    harness can load it without retraining.
# ---------------------------------------------------------------------------
from pathlib import Path
out = Path("checkpoints") / "smoke_model.keras"
out.parent.mkdir(parents=True, exist_ok=True)
model.save(str(out))
print(f"Saved {out}")
