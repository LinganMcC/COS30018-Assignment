from typing import Callable, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

InputShape = Tuple[int, int, int]
DEFAULT_INPUT_SHAPE: InputShape = (28, 28, 1)

# Registry: architecture name -> builder function. Filled by @_register below.
_BUILDERS: Dict[str, Callable[..., tf.keras.Model]] = {}

# Default head-dropout for each architecture (matches the Sprint 2 runs), used
# when the caller does not pass an explicit dropout value.
DEFAULT_DROPOUT: Dict[str, float] = {
    "mlp": 0.3,
    "lenet": 0.0,
    "vgg_deep": 0.5,
    "resnet": 0.3,
}


def _register(name: str):
    """Decorator that adds a builder function to the registry under `name`."""
    def wrapper(fn: Callable[..., tf.keras.Model]):
        _BUILDERS[name] = fn
        return fn
    return wrapper



# Shared building blocks


def _vgg_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Conv-BN-ReLU x2 -> MaxPool -> Dropout(0.25). Same block as Sprint 2."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    return x


def _residual_block(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    """Basic ResNet block; projects the shortcut when the shape changes."""
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([y, shortcut])
    return layers.ReLU()(out)


def _dense_head(x: tf.Tensor, units: int, dropout: float) -> tf.Tensor:
    """Dense-BN-ReLU-Dropout head used by the VGG models."""
    x = layers.Dense(units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


# Architectures


@_register("mlp")
def build_mlp(num_classes: int, dropout: float,
              input_shape: InputShape = DEFAULT_INPUT_SHAPE) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    for units in (1024, 512):
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="MLP_BN")


@_register("lenet")
def build_lenet(num_classes: int, dropout: float,
                input_shape: InputShape = DEFAULT_INPUT_SHAPE) -> tf.keras.Model:
    """Modernised LeNet-5: 6@5x5 -> pool -> 16@5x5 -> pool -> 120 -> 84."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(6, 5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(16, 5, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(84, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="LeNet5_modernised")


@_register("vgg_deep")
def build_vgg_deep(num_classes: int, dropout: float,
                   input_shape: InputShape = DEFAULT_INPUT_SHAPE) -> tf.keras.Model:
    """Deeper VGG: three conv blocks (32, 64, 128) + Dense(256) head."""
    inputs = layers.Input(shape=input_shape)
    x = _vgg_block(inputs, 32)   # 28 -> 14
    x = _vgg_block(x, 64)        # 14 -> 7
    x = _vgg_block(x, 128)       # 7  -> 3
    x = layers.Flatten()(x)
    x = _dense_head(x, 256, dropout)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="VGG_deep")


@_register("resnet")
def build_resnet(num_classes: int, dropout: float,
                 input_shape: InputShape = DEFAULT_INPUT_SHAPE) -> tf.keras.Model:
    """Small ResNet: stem + 2 blocks @32 + 2 blocks @64 + global average pool."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = _residual_block(x, 32)
    x = _residual_block(x, 32)
    x = _residual_block(x, 64, stride=2)   # 28 -> 14
    x = _residual_block(x, 64)
    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="Small_ResNet")


# Public API


def available_architectures() -> list:
    """Names accepted by build_model(), e.g. for a GUI drop-down."""
    return sorted(_BUILDERS)


def build_model(arch: str, num_classes: int = 10, dropout: float | None = None,
                input_shape: InputShape = DEFAULT_INPUT_SHAPE) -> tf.keras.Model:
    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from {available_architectures()}")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if dropout is None:
        dropout = DEFAULT_DROPOUT[arch]
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return _BUILDERS[arch](num_classes=num_classes, dropout=dropout,
                           input_shape=input_shape)
