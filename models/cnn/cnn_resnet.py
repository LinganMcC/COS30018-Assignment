import time
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from mnist_loader import load_mnist
from experiment_logger import log_run


RUN_ID = "cnn_resnet_run1"
ARCHITECTURE = "Small ResNet (residual blocks + BatchNorm)"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 30   # Shared budget across all models for a fair comparison.
DROPOUT = 0.3     # Head dropout before the classifier.
PATIENCE = 5      # EarlyStopping patience; shared across all models.

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "cnn_resnet.keras"
HISTORY_PLOT_PATH = Path(__file__).parent.parent / "experiments" / "cnn_resnet_history.png"


def residual_block(x, filters: int, stride: int = 1) -> tf.Tensor:
    shortcut = x

    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Project the shortcut if the shapes don't line up.
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([y, shortcut])
    out = layers.ReLU()(out)
    return out


def build_resnet() -> tf.keras.Model:
    inputs = layers.Input(shape=(28, 28, 1))

    # Stem.
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1: two residual blocks at 32 filters, 28x28.
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # Stage 2: downsample to 14x14 and widen to 64 filters.
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    # Classifier head.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Small_ResNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history: tf.keras.callbacks.History, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history.history["loss"], label="train")
    ax_loss.plot(history.history["val_loss"], label="val")
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
    ax_acc.plot(history.history["accuracy"], label="train")
    ax_acc.plot(history.history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy"); ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.legend(); ax_acc.grid(True, alpha=0.3)
    fig.suptitle(f"{ARCHITECTURE} - {RUN_ID}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def main() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Run ID            : {RUN_ID}")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print(f"Train : {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    model = build_resnet()
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_accuracy",
                                  save_best_only=True, verbose=1),
        # Shared with the other models: halve the LR when val_accuracy plateaus.
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                    patience=2, min_lr=1e-5, verbose=1),
    ]

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

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    val_acc = max(history.history["val_accuracy"])
    epochs_actually_run = len(history.history["loss"])

    print()
    print(f"Best val accuracy : {val_acc * 100:.2f} %")
    print(f"Test accuracy     : {test_acc * 100:.2f} %")
    print(f"Test loss         : {test_loss:.4f}")
    print(f"Epochs actually run: {epochs_actually_run} / {MAX_EPOCHS}")
    print(f"Training time     : {training_time_s:.1f} s")

    save_history_plot(history, HISTORY_PLOT_PATH)

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
        notes=f"Residual blocks + BN + dropout + LR-plateau; stopped at {epochs_actually_run}/{MAX_EPOCHS}.",
    )


if __name__ == "__main__":
    main()
