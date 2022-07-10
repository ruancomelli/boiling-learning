from typing import Optional, Tuple, Union

from tensorflow.keras.experimental import LinearModel as _LinearModel
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
    predictions = Dense(1, activation='linear')(inputs)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    x = AveragePooling2D((10, 10))(x)

    if normalize_images:
        x = ImageNormalization()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        16,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(32, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        16,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Conv2D(
        64,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Conv2D(
        64,
        (5, 5),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


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
    inputs = Input(shape=input_shape + (1,) if len(input_shape) == 2 else input_shape)
    x = inputs

    if normalize_images:
        x = ImageNormalization()(x)

    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(inputs)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(
        128,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Conv2D(
        128,
        (3, 3),
        padding='same',
        dtype=hidden_layers_policy,
    )(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(256, dtype=hidden_layers_policy)(x)
    x = ReLU()(x)
    if dropout is not None:
        x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    if problem == 'classification':
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem == 'regression':
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


def linear_model(
    input_shape: Tuple[int, ...],
    problem: Literal['classification', 'regression'] = 'regression',
    normalize_images: bool = True,
) -> ModelArchitecture:
    inputs = Input(shape=input_shape)
    x = inputs  # start "current layer" as the input layer
    if normalize_images:
        x = ImageNormalization()(x)

    if problem != 'regression':
        raise ValueError(f'unsupported problem type: \"{problem}\"')

    predictions = _LinearModel()(x)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)
