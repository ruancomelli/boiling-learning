from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPool2D

import boiling_learning as bl
from boiling_learning.management import ModelCreator

def is_classification(problem):
    return problem.lower() in {'classification', 'regime'}

def is_regression(problem):
    return problem.lower() in {'regression', 'heat flux', 'h', 'power'}

# CNN #1 implemented according to the paper Hobold and da Silva (2019): Visualization-based nucleate boiling heat flux quantification using machine learning.
def build(
    input_shape,
    dropout_ratio,
    problem='regression',
    num_classes=None,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(input_data)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    x = Dropout(dropout_ratio)(x)
    x = Flatten()(x)
    x = Dense(200, activation='relu')(x)
    x = Dropout(dropout_ratio)(x)

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
    verbose,
    checkpoint,
    dropout_ratio,
    num_classes,
    problem,
    compile_setup,
    fit_setup,
    fetch,
):    
    last_epoch, model = bl.model.restore(**checkpoint)
    initial_epoch = max(last_epoch, 0)
    
    if model is None:        
        model = build(
            input_shape,
            dropout_ratio,
            problem,
            num_classes
        )

        if compile_setup.get('do', False):
            model.compile(**compile_setup['params'])

    history = None
    if fit_setup.get('do', False):
        fit_setup['params']['initial_epoch'] = initial_epoch
        history = model.fit(**fit_setup['params'])

    available_data = {
        'model': model,
        'history': history
    }

    return {
        k: available_data[k]
        for k in fetch
    }

creator = ModelCreator(
    creator_method=creator_method,
    creator_name='HoboldNet1',
    default_params=dict(
        input_shape=[224, 224, 1],
        verbose=0,
        checkpoint={'restore': False},
        dropout_ratio=0.5,
        num_classes=3,
        problem='regression',
        fetch=['model', 'history'],
    ),
    expand_params=True
)