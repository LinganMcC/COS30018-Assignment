import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from mnist_loader import load_mnist
from experiment_logger import log_run


RUN_ID = "mlp_simple_run2"
ARCHITECTURE = "MLP (1024-512, BatchNorm + augmentation)"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 30   # Shared budget across all models for a fair comparison.
DROPOUT = 0.3     # Dropout between the dense layers.
# Run 1 (256-128, no BN, no augmentation) overfit: val loss flattened at ~0.07
# while train loss kept falling. A small sweep picked this setup on val loss.
HIDDEN_UNITS = (1024, 512)
AUG_SHIFT = 0.08  # Random shift up to 8% (~2 px) - an MLP has no built-in shift tolerance.
AUG_ROTATE = 0.03 # Random rotation up to ~11 degrees.
SEED = 42
PATIENCE = 5      # EarlyStopping patience; shared across all models.

MODEL_PATH = Path(__file__).parent.parent / "checkpoints" / "mlp_simple.keras"
HISTORY_PLOT_PATH = Path(__file__).parent.parent / "experiments" / "mlp_simple_history.png"
HISTORY_JSON_PATH = Path(__file__).parent.parent / "experiments" / "mlp_simple_history.json"


def build_mlp() -> tf.keras.Model:
    inputs = layers.Input(shape=(28, 28, 1))
    # Augmentation layers are only active during training; at inference they pass through.
    x = layers.RandomTranslation(AUG_SHIFT, AUG_SHIFT, fill_mode="constant")(inputs)
    x = layers.RandomRotation(AUG_ROTATE, fill_mode="constant")(x)
    # Flatten the image into a 784-vector; no convolutions here.
    x = layers.Flatten()(x)
    for units in HIDDEN_UNITS:
        # Dense -> BatchNorm -> ReLU -> Dropout. BN keeps activations well scaled,
        # which smooths training; bias is redundant before BN.
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="MLP_BN_Aug")
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

    tf.keras.utils.set_random_seed(SEED)  # Reproducible weights and augmentation.
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print(f"Train : {x_train.shape}  Val: {x_val.shape}  Test: {x_test.shape}")

    model = build_mlp()
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Monitor val_loss rather than val_accuracy: at ~99% accuracy, val_accuracy moves
    # in noisy 0.01% steps, so EarlyStopping/LR decay react to noise. Loss is smoother.
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=str(MODEL_PATH), monitor="val_loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
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
    # Save raw per-epoch history so compare_models.py can overlay all models.
    HISTORY_JSON_PATH.write_text(json.dumps(
        {k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2))

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
        notes=f"MLP 1024-512 + BN + shift/rotate aug; monitor val_loss; stopped at {epochs_actually_run}/{MAX_EPOCHS}.",
    )


if __name__ == "__main__":
    main()
