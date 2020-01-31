from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D

import utils
import management

def is_classification(problem):
    return problem.lower() in ['classification', 'regime']

def is_regression(problem):
    return problem.lower() in ['regression', 'heat flux', 'h', 'power']

def build(
    input_shape,
    problem,
    num_classes,
):
    input_data = Input(shape=input_shape)
    x = Conv2D(64,(3, 3), padding='same', activation='relu')(input_data) 
    x = Conv2D(64,(3, 3), padding='same', activation='relu')(x) 
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64,(3, 3), padding='same', activation='relu')(x) 
    x = Conv2D(64,(3, 3), padding='same', activation='relu')(x) 
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128,(3, 3), padding='same', activation='relu')(x) 
    x = Conv2D(128,(3, 3), padding='same', activation='relu')(x) 
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    
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
        if fit_setup['method'] == 'fit':
            history = model.fit(**fit_setup['params'])
        elif fit_setup['method'] == 'fit_generator':
            history = model.fit_generator(**fit_setup['params'])
    
    return {
        'model': model,
        'history': history
    }
    
creator = management.ModelCreator(
    creator_method=creator_method,
    creator_name='kramer_net',
    default_params=dict(
        input_shape=(224, 224, 3),
        num_classes=3,
        problem='regression',
    ),
    expand_params=True
)