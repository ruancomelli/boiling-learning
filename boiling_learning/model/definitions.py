from typing import Optional, Tuple, Union

from tensorflow.keras.experimental import LinearModel
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
    ReLU,
    Softmax,
)
from tensorflow.keras.mixed_precision.experimental import Policy
from typing_extensions import Literal

from boiling_learning.model.layers import ImageNormalization
from boiling_learning.model.model import ModelArchitecture

# Check this guideline:
# https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
# It includes tips and rules-of-thumb for defining layers.


def linear_regression(input_shape: Tuple[int, ...]) -> ModelArchitecture:
    inputs = Input(shape=input_shape)
    outputs = Dense(1)(inputs)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def tiny_convnet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    outputs = AveragePooling2D((10, 10))(outputs)

    if normalize_images:
        outputs = ImageNormalization()(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)
    outputs = Flatten(dtype=hidden_layers_policy)(outputs)

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def small_convnet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        16,
        (5, 5),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet1(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    '''CNN #1 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        16,
        (5, 5),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet2(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    '''CNN #2 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet3(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    '''CNN #3 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (5, 5),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def hoboldnet_supplementary(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    '''See supplementary material for Hobold and da Silva (2019): Visualization-based
    nucleate boiling heat flux quantification using machine learning.
    '''
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        32,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (5, 5),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def kramernet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: Literal['classification', 'regression'] = 'regression',
    num_classes: Optional[int] = None,
    normalize_images: bool = True,
) -> ModelArchitecture:
    '''See supplementary material for Hobold and da Silva (2019): Visualization-based
    nucleate boiling heat flux quantification using machine learning.
    '''
    # start "current layer" as the input layer
    outputs = inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)

    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    if dropout is not None:
        outputs = Dropout(dropout, dtype=hidden_layers_policy)(outputs)

    outputs = Conv2D(
        128,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(outputs)
    outputs = ReLU()(outputs)
    outputs = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(outputs)
    outputs = Conv2D(
        128,
        (3, 3),
        padding='same',
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

    if problem == 'classification':
        outputs = Dense(num_classes, dtype=hidden_layers_policy)(outputs)
        outputs = Softmax(dtype=output_layer_policy)(outputs)
    elif problem == 'regression':
        outputs = Dense(1, dtype=hidden_layers_policy)(outputs)
        outputs = Activation('linear', dtype=output_layer_policy)(outputs)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)


def linear_model(
    input_shape: Tuple[int, ...],
    normalize_images: bool = True,
) -> ModelArchitecture:
    outputs = inputs = Input(shape=input_shape)
    if normalize_images:
        outputs = ImageNormalization()(outputs)

    outputs = LinearModel()(outputs)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs)
