# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras.models import Model

# from boiling_learning.model.model import make_creator_method
# from boiling_learning.preprocessing.transformers import Creator


# def build(
#     input_shape,
# ) -> Model:
#     input_data = Input(shape=input_shape)
#     predictions = Dense(1, activation='linear')(input_data)

#     return Model(inputs=input_data, outputs=predictions)


# creator = Creator('LinearRegression', make_creator_method(builder=build))


# # creator = ElementCreator(
# #     method=make_creator_method(builder=build),
# #     name='LinearRegression',
# #     expand_params=True
# # )
