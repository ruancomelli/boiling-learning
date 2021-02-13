from typing import (
    Optional,
    Tuple,
    Union
)

from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D
)
from tensorflow.keras.mixed_precision.experimental import Policy
from tensorflow.keras.models import Model

from boiling_learning.utils.functional import (
    pack
)
from boiling_learning.model.model import (
    ProblemType,
    make_creator
)


'''
Check this guideline: https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
It includes tips and rules-of-thumb for defining layers.
'''


@make_creator('LinearRegression')
def LinearRegression(input_shape: Tuple, **kwargs) -> Model:
    input_data = Input(shape=input_shape)
    predictions = Dense(1, activation='linear')(input_data)

    model = Model(inputs=input_data, outputs=predictions)

    return model


@make_creator(
    'HoboldNet1',
    defaults=pack(
        verbose=2,
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=frozenset({'model', 'history'})
    )
)
def HoboldNet1(
        input_shape: Tuple,
        dropout: Optional[float],
        hidden_layers_policy: Union[str, Policy],
        output_layer_policy: Union[str, Policy],
        problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
        num_classes: Optional[int] = None
) -> Model:
    '''CNN #1 implemented according to the paper Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning.
    '''
    input_data = Input(shape=input_shape)
    x = Conv2D(16, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = utils.elem_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model


@make_creator(
    'HoboldNet2',
    defaults=pack(
        verbose=2,
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=frozenset({'model', 'history'}),
    )
)
def HoboldNet2(
        input_shape: Tuple,
        dropout: Optional[float],
        hidden_layers_policy: Union[str, Policy],
        output_layer_policy: Union[str, Policy],
        problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
        num_classes: Optional[int] = None
) -> Model:
    '''CNN #2 implemented according to the paper Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning.
    '''
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = utils.elem_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model


@make_creator(
    'HoboldNet3',
    defaults=pack(
        verbose=2,
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=frozenset({'model', 'history'}),
    )
)
def HoboldNet3(
        input_shape: Tuple,
        dropout: Optional[float],
        hidden_layers_policy: Union[str, Policy],
        output_layer_policy: Union[str, Policy],
        problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
        num_classes: Optional[int] = None
) -> Model:
    '''CNN #3 implemented according to the paper Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning.
    '''
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(64, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(200, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = utils.elem_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model


@make_creator(
    'HoboldNetSupplementary',
    defaults=pack(
        verbose=2,
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=frozenset({'model', 'history'})
    )
)
def HoboldNetSupplementary(
        input_shape: Tuple,
        dropout: Optional[float],
        hidden_layers_policy: Union[str, Policy],
        output_layer_policy: Union[str, Policy],
        problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
        num_classes: Optional[int] = None
) -> Model:
    '''See supplementary material for Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning
    '''
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(64, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = utils.elem_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model


@make_creator(
    'KramerNet',
    defaults=pack(
        verbose=2,
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=frozenset({'model', 'history'})
    )
)
def KramerNet(
        input_shape: Tuple,
        dropout: Optional[float],
        hidden_layers_policy: Union[str, Policy],
        output_layer_policy: Union[str, Policy],
        problem: Union[int, str, ProblemType] = ProblemType.REGRESSION,
        num_classes: Optional[int] = None
):
    input_data = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(256, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout, dtype=hidden_layers_policy)(x)

    problem = utils.elem_item(ProblemType, problem)
    if problem is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif problem is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model
