import time
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from models.cnn.mnist_loader import load_mnist
from models.cnn.experiment_logger import log_run


RUN_ID = "cnn_lenet_run1"
ARCHITECTURE = "LeNet-5 (modernised)"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 20
DROPOUT = 0.0     # LeNet-5 baseline uses no dropout; variant 3 will add it.
PATIENCE = 3      # EarlyStopping patience on validation accuracy.

MODEL_PATH = Path("checkpoints") / "cnn_lenet.keras"
HISTORY_PLOT_PATH = Path("experiments") / "cnn_lenet_history.png"



def build_lenet5() -> tf.keras.Model:
    """Build the modernised LeNet-5 model described in this file's docstring."""
    model = models.Sequential(name="LeNet5_modernised")
    model.add(layers.Input(shape=(28, 28, 1)))

    # Block 1: 6 filters of 5x5, then 2x2 max-pool.
    model.add(layers.Conv2D(6, kernel_size=5, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Block 2: 16 filters of 5x5, then 2x2 max-pool. No padding here, matching
    # the original 1998 architecture, so we drop from 14x14 -> 10x10 -> 5x5.
    model.add(layers.Conv2D(16, kernel_size=5, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=2))

    # Classifier head.
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history: tf.keras.callbacks.History, out_path: Path) -> None:
    """Save training/validation loss + accuracy curves to a PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(f"{ARCHITECTURE} - {RUN_ID}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {out_path}")



def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Run ID            : {RUN_ID}")

    # 1. Load MNIST through the shared loader (same split every run).
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print(f"Train : {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    # 2. Build the model.
    model = build_lenet5()
    model.summary()

    # 3. Set up callbacks:
    #    * EarlyStopping to avoid overfitting past the validation peak.
    #    * ModelCheckpoint to keep the best-val-accuracy weights on disk.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 4. Train.
    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=2,
    )
    training_time_s = time.time() - start

    # 5. Evaluate ONCE on the held-out test set.
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    val_acc = max(history.history["val_accuracy"])
    epochs_actually_run = len(history.history["loss"])

    print()
    print(f"Best val accuracy : {val_acc * 100:.2f} %")
    print(f"Test accuracy     : {test_acc * 100:.2f} %")
    print(f"Test loss         : {test_loss:.4f}")
    print(f"Epochs actually run: {epochs_actually_run} / {MAX_EPOCHS}")
    print(f"Training time     : {training_time_s:.1f} s")

    # 6. Save the training-curve plot for the report.
    save_history_plot(history, HISTORY_PLOT_PATH)

    # 7. Append this run to the shared experiment log.
    log_run(
        run_id=RUN_ID,
        architecture=ARCHITECTURE,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=epochs_actually_run,
        dropout=DROPOUT,
        val_accuracy=val_acc,
        test_accuracy=test_acc,
        training_time_s=training_time_s,
        notes=f"Baseline; EarlyStopping stopped after {epochs_actually_run}/{MAX_EPOCHS} epochs.",
    )


if __name__ == "__main__":
    main()
