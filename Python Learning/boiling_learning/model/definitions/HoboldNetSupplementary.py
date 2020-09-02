from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D

import boiling_learning as bl
from boiling_learning.management import ElementCreator
from boiling_learning.model.definitions.utils import (
    make_creator_method,
    ProblemType,
)

# See supplementary material for Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning
def build(
    input_shape,
    dropout_ratio,
    hidden_layers_policy,
    output_layer_policy,
    problem=ProblemType.REGRESSION,
    num_classes=None,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(input_data)
    x = Conv2D(64, (5, 5), padding='same', activation='relu', dtype=hidden_layers_policy)(x)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Dropout(dropout_ratio, dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dense(512, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout_ratio, dtype=hidden_layers_policy)(x)
    
    if ProblemType.get_type(problem) is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif ProblemType.get_type(problem) is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model

creator = ElementCreator(
    creator_method=make_creator_method(builder=build),
    creator_name='HoboldNetSupplementary',
    default_params=dict(
        verbose=2,
        checkpoint={'restore': False},
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=['model', 'history'],
    ),
    expand_params=True
)
