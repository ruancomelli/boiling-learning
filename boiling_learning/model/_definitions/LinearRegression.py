from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

from boiling_learning.preprocessing.transformers import Creator
from boiling_learning.management import ElementCreator
from boiling_learning.model.model import (
    make_creator_method
)


def build(
    input_shape,
):
    input_data = Input(shape=input_shape)
    predictions = Dense(1, activation='linear')(input_data)

    model = Model(inputs=input_data, outputs=predictions)

    return model


creator = Creator(
    'LinearRegression',
    make_creator_method(builder=build)    
)


# creator = ElementCreator(
#     method=make_creator_method(builder=build),
#     name='LinearRegression',
#     expand_params=True
# )
