from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model

from boiling_learning.management import ElementCreator
from boiling_learning.model.model import ProblemType, make_creator_method


def build(
    input_shape,
    hidden_layers_policy,
    output_layer_policy,
    problem=ProblemType.REGRESSION,
    num_classes=None,
):
    mobile_net = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    x = Dense(256, activation='relu', dtype=hidden_layers_policy)(mobile_net.output)
    
    if ProblemType.get_type(problem) is ProblemType.CLASSIFICATION:
        x = Dense(num_classes, dtype=hidden_layers_policy)(x)
        predictions = Activation('softmax', dtype=output_layer_policy)(x)
    elif ProblemType.get_type(problem) is ProblemType.REGRESSION:
        x = Dense(1, dtype=hidden_layers_policy)(x)
        predictions = Activation('linear', dtype=output_layer_policy)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=mobile_net.input, outputs=predictions)

    return model


creator = ElementCreator(
    method=make_creator_method(builder=build),
    name='BLMobileNet',
    default_params=dict(
        verbose=2,
        checkpoint={'restore': False},
        num_classes=None,
        problem=ProblemType.REGRESSION,
        fetch=['model', 'history'],
    ),
    expand_params=True
)
