from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

import boiling_learning as bl
from boiling_learning.management import ElementCreator
from boiling_learning.model.definitions.utils import (
    make_creator_method
)

def build(
    input_shape,
):
    input_data = Input(shape=input_shape)
    predictions = Dense(1, activation='linear')(input_data)

    model = Model(inputs=input_data, outputs=predictions)
    
    return model
    
creator = ElementCreator(
    creator_method=make_creator_method(builder=build),
    creator_name='LinearRegression',
    expand_params=True
)
