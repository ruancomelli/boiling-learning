import enum
from functools import partial
from typing import Optional, Tuple, Union

import funcy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.experimental import LinearModel as _LinearModel
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Lambda,
    Layer,
    MaxPool2D,
    ReLU,
    SeparableConv2D,
    Softmax,
    SpatialDropout2D,
    TimeDistributed,
)
from tensorflow.keras.mixed_precision.experimental import Policy

from boiling_learning.model.layers import ImageNormalization
from boiling_learning.model.model import ModelArchitecture, ProblemType

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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
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
    problem: ProblemType = ProblemType.REGRESSION,
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

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


def linear_model(
    input_shape: Tuple[int, ...],
    problem: ProblemType = ProblemType.REGRESSION,
    normalize_images: bool = True,
) -> ModelArchitecture:
    inputs = Input(shape=input_shape)
    x = inputs  # start "current layer" as the input layer
    if normalize_images:
        x = ImageNormalization()(x)

    if problem is not ProblemType.REGRESSION:
        raise ValueError(f'unsupported problem type: \"{problem}\"')

    predictions = _LinearModel()(x)

    return ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=predictions)


class FlatteningMode(enum.Enum):
    FLATTEN = enum.auto()
    AVERAGE_POOLING = enum.auto()
    MAX_POOLING = enum.auto()


class ConvolutionType(enum.Enum):
    CONV = enum.auto()
    SEPARABLE_CONV = enum.auto()


def boilnet(
    image_shape: Tuple[Optional[int], ...],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    dropout: Optional[float] = None,
    spatial_dropout: Optional[float] = None,
    time_window: int = 0,
    convolution_type: ConvolutionType = ConvolutionType.CONV,
    flattening: FlatteningMode = FlatteningMode.FLATTEN,
    problem: ProblemType = ProblemType.REGRESSION,
    num_classes: int = 0,
    normalize_images: bool = True,
) -> ModelArchitecture:
    input_shape = (time_window, *image_shape) if time_window > 0 else image_shape
    flatten_layer = {
        FlatteningMode.FLATTEN: Flatten,
        FlatteningMode.AVERAGE_POOLING: GlobalAveragePooling2D,
        FlatteningMode.MAX_POOLING: GlobalMaxPooling2D,
    }[flattening]

    inputs = Input(shape=input_shape)

    normalized = ImageNormalization()(inputs) if normalize_images else inputs
    distribute = TimeDistributed if time_window > 0 else funcy.identity
    if spatial_dropout is not None:
        spatial_dropouter = partial(SpatialDropout2D, spatial_dropout)
    else:
        spatial_dropouter = funcy.constantly(Layer())

    conv_layer = {
        ConvolutionType.CONV: Conv2D,
        ConvolutionType.SEPARABLE_CONV: SeparableConv2D,
    }[convolution_type]

    conv = distribute(conv_layer(32, (5, 5), padding='same', activation='relu'))(normalized)
    conv = distribute(spatial_dropouter())(conv)
    conv = distribute(MaxPool2D((2, 2), strides=(2, 2)))(conv)
    conv = distribute(conv_layer(64, (5, 5), padding='same', activation='relu'))(conv)
    conv = distribute(spatial_dropouter())(conv)
    conv = distribute(MaxPool2D((2, 2), strides=(2, 2)))(conv)
    flatten = distribute(flatten_layer())(conv)

    if dropout is not None:
        dropouter = partial(Dropout, dropout)
    else:
        dropouter = funcy.constantly(Lambda(funcy.identity))

    head = dropouter()(flatten)
    head = Dense(256)(head)
    head = ReLU()(head)
    head = dropouter()(head)

    head_size, activation = {
        ProblemType.CLASSIFICATION: (num_classes, 'softmax'),
        ProblemType.REGRESSION: (1, 'linear'),
    }[problem]
    outputs = Dense(head_size, activation=activation)(flatten)

    return _apply_policies_to_layers(
        ModelArchitecture.from_inputs_and_outputs(inputs=inputs, outputs=outputs),
        hidden_layers_policy,
        output_layer_policy,
    )


def _apply_policies_to_layers(
    model: ModelArchitecture,
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
) -> ModelArchitecture:
    hidden_layers_policy = Policy(hidden_layers_policy)
    output_layer_policy = Policy(output_layer_policy)

    for layer in model.model.layers[1:-1]:
        layer.dtype = hidden_layers_policy
    model.model.layers[-1].dtype = output_layer_policy

    return model


def boiling_mobile_net(
    image_shape: Tuple[Optional[int], ...],
    hidden_layers_policy: Optional[Union[str, Policy]] = None,
    output_layer_policy: Optional[Union[str, Policy]] = None,
    problem: ProblemType = ProblemType.REGRESSION,
    num_classes: int = 0,
) -> ModelArchitecture:
    mobile_net = MobileNetV2(
        input_shape=image_shape, include_top=False, weights='imagenet', pooling='avg'
    )
    x = Dense(256, dtype=hidden_layers_policy)(mobile_net.output)
    x = ReLU()(x)

    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Softmax(dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return ModelArchitecture.from_inputs_and_outputs(inputs=mobile_net.input, outputs=predictions)


# TODO: fazer erro em função do y: ver se para maiores ys o erro vai subindo ou diminuindo
# quem sabe fazer 3 ou mais modelos, um especializado para cada região de y; e quem sabe
# usar um classificador pra escolher qual estimador não ajude muito
# focar na arquitetura da rede, que é mais importante do que hiperparâmetros
# otimizar as convolucionais pode ser mais importante do que otimizar as fully-connected

# TODO: ReLU or LeakyReLU? https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning
