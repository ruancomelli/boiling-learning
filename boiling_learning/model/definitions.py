from typing import Literal

from tensorflow.keras.experimental import LinearModel
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPool2D,
    ReLU,
    Softmax,
)
from tensorflow.keras.mixed_precision import Policy
from typing_extensions import assert_never

from boiling_learning.model.layers import ImageNormalization
from boiling_learning.model.model import ModelArchitecture

# Check this guideline:
# https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
# It includes tips and rules-of-thumb for defining layers.


def linear_regression(input_shape: tuple[int, ...]) -> ModelArchitecture:
    inputs = Input(shape=input_shape)
    outputs = Dense(1)(inputs)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def tiny_convnet(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    outputs = AveragePooling2D((10, 10))(outputs)

    if normalize_images:
        outputs = ImageNormalization()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def small_convnet(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        16,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(32, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet1(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    """Instantiate the first model from Hobold and da Silva (2019).

    This function instantiates the model named "CNN #1" in the paper Hobold and
    da Silva (2019): Visualization-based nucleate boiling heat flux quantification
    using machine learning.
    """
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        16,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(200, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet2(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    """Instantiate the second model from Hobold and da Silva (2019).

    This function instantiates the model named "CNN #2" in the paper Hobold and
    da Silva (2019): Visualization-based nucleate boiling heat flux quantification
    using machine learning.
    """
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(200, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet3(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    """Instantiate the third model from Hobold and da Silva (2019).

    This function instantiates the model named "CNN #3" in the paper Hobold and
    da Silva (2019): Visualization-based nucleate boiling heat flux quantification
    using machine learning.
    """
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(512, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet_supplementary(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    """Instantiate the supplementary model from Hobold and da Silva (2019).

    See supplementary material for Hobold and da Silva (2019): Visualization-based
    nucleate boiling heat flux quantification using machine learning.
    """
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (5, 5),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(512, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def kramernet(
    input_shape: tuple[int, int, int] | tuple[int, int],
    dropout: float | None,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
    problem: Literal["classification", "regression"] = "regression",
    num_classes: int | None = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    """Instantiate the model from Scariot (2018)."""
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=_ensure_3d_input(input_shape))

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        64,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = Conv2D(
        64,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = Conv2D(
        128,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        128,
        (3, 3),
        padding="same",
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = Flatten(dtype=hidden_layers_policy)(outputs)
    outputs = Dense(256, dtype=hidden_layers_policy)(outputs)
    outputs = ReLU()(outputs)

    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = _append_output_layer(
        outputs,
        problem,
        num_classes,
        hidden_layers_policy=hidden_layers_policy,
        output_layer_policy=output_layer_policy,
    )

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def linear_model(
    input_shape: tuple[int, ...],
    normalize_images: bool = True,
) -> ModelArchitecture:
    outputs = inputs = Input(shape=input_shape)
    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = LinearModel()(outputs)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


# TODO: add two overloads:
# - one for regression, where `num_classes` is not forbidden
# - one for classification, where `num_classes` is required
def _append_output_layer(
    outputs: Layer,
    problem: Literal["classification", "regression"],
    num_classes: int | None = None,
    *,
    hidden_layers_policy: str | Policy | None = None,
    output_layer_policy: str | Policy | None = None,
) -> Layer:
    match problem:
        case "classification":
            outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
            outputs = Softmax(dtype=output_layer_policy)(outputs)
        case "regression":
            outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
            outputs = Activation("linear", dtype=output_layer_policy)(outputs)
        case never:
            assert_never(never)

    return outputs


def _ensure_3d_input(
    input_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[int, int, int]:
    """Ensure that the input shape is 3-dimensional.

    Expects a 2-dimensional input shape (height, width) and converts it to a
    3-dimensional input shape (height, width, 1) where the last dimension is
    the grayscale channel.
    """
    match input_shape:
        case (h, w):
            return (h, w, 1)
        case _:
            return input_shape
