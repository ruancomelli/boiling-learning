from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from boiling_learning.management import ModelCreator
import boiling_learning.utils

def build(
    input_shape,
):
    input_data = Input(shape=input_shape)
    predictions = Dense(1, activation='linear')(input_data)

    model = Model(inputs=input_data, outputs=predictions)
    
    return model

def creator_method(
    input_shape,
    compile_setup,
    fit_setup,
):      
    compile_setup, fit_setup = boiling_learning.utils.regularize_default(
        (compile_setup, fit_setup), 
        cond=lambda x: x is not None,
        default=lambda x: dict(do=False),
        many=True,
        call_default=True
    )
    
    model = build(
        input_shape,
    )
        
    if compile_setup['do']:
        model.compile(**compile_setup['params'])

    model.summary()

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
    
creator = ModelCreator(
    creator_method=creator_method,
    creator_name='linear_regression',
    expand_params=True
)