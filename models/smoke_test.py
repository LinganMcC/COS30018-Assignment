import time
import tensorflow as tf
from tensorflow.keras import layers, models


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise pixel values from 0-255 into 0.0-1.0.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"TensorFlow version : {tf.__version__}")
print(f"Training set shape : {x_train.shape}")
print(f"Test set shape     : {x_test.shape}")


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


start = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=2)
train_time = time.time() - start


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print()
print(f"Training time    : {train_time:.2f} s")
print(f"Test accuracy    : {test_acc * 100:.2f} %")
print(f"Test loss        : {test_loss:.4f}")


model.save("smoke_model.keras")
print("Saved smoke_model.keras")
