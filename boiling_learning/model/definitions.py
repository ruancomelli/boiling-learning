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
    LayerNormalization,
    MaxPool2D,
    SeparableConv2D,
    SpatialDropout2D,
    TimeDistributed,
)
from tensorflow.keras.mixed_precision.experimental import Policy

from boiling_learning.model.model import Model, ProblemType
from boiling_learning.utils import enum_item

# Check this guideline:
# https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
# It includes tips and rules-of-thumb for defining layers.


def linear_regression(input_shape: Tuple[int, ...]) -> Model:
    input_data = Input(shape=input_shape)
    predictions = Dense(1, activation='linear')(input_data)

    return Model(inputs=input_data, outputs=predictions)


def tiny_convnet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    x = AveragePooling2D((10, 10))(x)

    if normalize_images:
        x = LayerNormalization()(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def small_convnet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        16,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(32, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def hoboldnet1(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    '''CNN #1 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        16,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def hoboldnet2(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    '''CNN #2 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def hoboldnet3(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    '''CNN #3 implemented according to the paper Hobold and da Silva (2019):
    Visualization-based nucleate boiling heat flux quantification using machine
    learning.
    '''
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = Conv2D(
        64,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def hoboldnet_supplementary(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    '''See supplementary material for Hobold and da Silva (2019): Visualization-based
    nucleate boiling heat flux quantification using machine learning.
    '''
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        32,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = Conv2D(
        64,
        (5, 5),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def kramernet(
    input_shape: Union[Tuple[int, int, int], Tuple[int, int]],
    dropout: Optional[float],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: Optional[int] = None,
    normalize_images: bool = False,
) -> Model:
    '''See supplementary material for Hobold and da Silva (2019): Visualization-based
    nucleate boiling heat flux quantification using machine learning.
    '''
    if len(input_shape) == 2:
        input_shape = input_shape + (1,)

    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer

    if normalize_images:
        x = LayerNormalization()(x)

    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(input_data)
    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(input_data)
    x = Conv2D(
        64,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(
        128,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(input_data)
    x = Conv2D(
        128,
        (3, 3),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(256, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


def linear_model(
    input_shape: Tuple,
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    normalize_images: bool = False,
) -> Model:
    input_data = Input(shape=input_shape)
    x = input_data  # start "current layer" as the input layer
    if normalize_images:
        x = LayerNormalization()(x)

    problem = enum_item(ProblemType, problem.upper())
    if problem is not ProblemType.REGRESSION:
        raise ValueError(f'unsupported problem type: \"{problem}\"')

    predictions = _LinearModel()(x)

    return Model(inputs=input_data, outputs=predictions)


class FlatteningMode(enum.Enum):
    FLATTEN = enum.auto()
    AVERAGE_POOLING = enum.auto()
    MAX_POOLING = enum.auto()


class ConvolutionType(enum.Enum):
    CONV = enum.auto()
    SEPARABLE_CONV = enum.auto()


def boilnet(
    image_shape: Tuple[Optional[int], ...],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    dropout: Optional[float] = None,
    spatial_dropout: Optional[float] = None,
    time_window: int = 0,
    convolution_type: Union[ConvolutionType, str, int] = ConvolutionType.CONV,
    flattening: Union[FlatteningMode, str, int] = FlatteningMode.FLATTEN,
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: int = 0,
    normalize_images: bool = False,
) -> Model:
    input_shape = (time_window, *image_shape) if time_window > 0 else image_shape
    flattening = enum_item(FlatteningMode, flattening)
    flatten_layer = {
        FlatteningMode.FLATTEN: Flatten,
        FlatteningMode.AVERAGE_POOLING: GlobalAveragePooling2D,
        FlatteningMode.MAX_POOLING: GlobalMaxPooling2D,
    }[flattening]

    inputs = Input(shape=input_shape)

    normalized = LayerNormalization()(inputs) if normalize_images else inputs
    distribute = TimeDistributed if time_window > 0 else funcy.identity
    if spatial_dropout is not None:
        spatial_dropouter = partial(SpatialDropout2D, spatial_dropout)
    else:
        spatial_dropouter = funcy.constantly(Layer())

    convolution_type = enum_item(ConvolutionType, convolution_type)
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
    head = Dense(256, activation='relu')(head)
    head = dropouter()(head)

    problem = enum_item(ProblemType, problem)
    head_size, activation = {
        ProblemType.CLASSIFICATION: (num_classes, 'softmax'),
        ProblemType.REGRESSION: (1, 'linear'),
    }[problem]
    outputs = Dense(head_size, activation=activation)(flatten)

    model = Model(inputs=inputs, outputs=outputs)

    model = _apply_policies_to_layers(model, hidden_layers_policy, output_layer_policy)

    return model


def _apply_policies_to_layers(
    model: Model,
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
) -> Model:
    hidden_layers_policy = Policy(hidden_layers_policy)
    output_layer_policy = Policy(output_layer_policy)

    for layer in model.layers[1:-1]:
        layer.dtype = hidden_layers_policy
    model.layers[-1].dtype = output_layer_policy

    return model


def boiling_mobile_net(
    image_shape: Tuple[Optional[int], ...],
    hidden_layers_policy: Union[str, Policy],
    output_layer_policy: Union[str, Policy],
    problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
    num_classes: int = 0,
) -> Model:
    mobile_net = MobileNetV2(
        input_shape=image_shape, include_top=False, weights='imagenet', pooling='avg'
    )
    x = Dense(256, activation='relu', dtype=hidden_layers_policy)(mobile_net.output)

    problem = enum_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=mobile_net.input, outputs=predictions)


# fazer erro em função do y: ver se para maiores ys o erro vai subindo ou diminuindo
# quem sabe fazer 3 ou mais modelos, um especializado para cada região de y; e quem sabe
# usar um classificador pra escolher qual estimador não ajude muito
# focar na arquitetura da rede, que é mais importante do que hiperparâmetros
# otimizar as convolucionais pode ser mais importante do que otimizar as fully-connected
