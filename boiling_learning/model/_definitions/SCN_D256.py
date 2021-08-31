from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)
from tensorflow.keras.models import Model

from boiling_learning.management import ElementCreator
from boiling_learning.model.model import ProblemType, make_creator_method


# SCN_D256 :: SingleConvNet with a 256-unit Dense layer
def build(
    input_shape,
    dropout_ratio,
    hidden_layers_policy,
    output_layer_policy,
    conv2d_stride: int = 5,
    problem=ProblemType.REGRESSION,
    num_classes=None,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(
        32,
        (conv2d_stride, conv2d_stride),
        padding='same',
        activation='relu',
        dtype=hidden_layers_policy,
    )(input_data)
    x = MaxPool2D((2, 2), strides=(2, 2), dtype=hidden_layers_policy)(x)
    x = Flatten(dtype=hidden_layers_policy)(x)
    x = Dropout(dropout_ratio, dtype=hidden_layers_policy)(x)
    x = Dense(256, activation='relu', dtype=hidden_layers_policy)(x)
    x = Dropout(dropout_ratio, dtype=hidden_layers_policy)(x)

    if ProblemType.get_type(problem) is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif ProblemType.get_type(problem) is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    return Model(inputs=input_data, outputs=predictions)


creator = ElementCreator(
    method=make_creator_method(builder=build),
    name='SCN_D256',
    default_params=dict(
        verbose=2,
        checkpoint={'restore': False},
        num_classes=3,
        problem=ProblemType.REGRESSION,
        fetch=['model', 'history'],
    ),
    expand_params=True,
)
