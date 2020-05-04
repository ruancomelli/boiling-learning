from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D

import utils
import management

def is_classification(problem):
    return problem.lower() in {'classification', 'regime'}

def is_regression(problem):
    return problem.lower() in {'regression', 'heat flux', 'h', 'power'}

# See supplementary material for Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning
def build(
    input_shape,
    problem='regression',
    num_classes=None,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), padding='same', activation='relu')(input_data)
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.5)(x)

    if is_classification(problem):
        predictions = Dense(num_classes, activation='softmax')(x)
    elif is_regression(problem):
        predictions = Dense(1)(x)
    else:
        raise ValueError(f'unknown problem type: \"{problem}\"')

    model = Model(inputs=input_data, outputs=predictions)

    return model

def creator_method(
    input_shape,
    num_classes,
    problem,
    compile_setup,
    fit_setup,
    fetch,
):
    compile_setup, fit_setup = utils.regularize_default(
        (compile_setup, fit_setup),
        cond=lambda x: x is not None,
        default=lambda x: dict(do=False),
        many=True,
        call_default=True
    )

    model = build(
        input_shape,
        problem,
        num_classes
    )

    if compile_setup['do']:
        model.compile(**compile_setup['params'])

    history = None
    if fit_setup['do']:
        history = model.fit(**fit_setup['params'])

    available_data = {
        'model': model,
        'history': history
    }

    return {
        k: available_data[k]
        for k in fetch
    }

creator = management.ModelCreator(
    creator_method=creator_method,
    creator_name='HoboldNetSupplementary',
    default_params=dict(
        input_shape=[224, 224, 1],
        num_classes=3,
        problem='regression',
    ),
    expand_params=True
)