import os
import pathlib

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import boiling_learning.utils
import management
from management import Mirror

from model_definitions import KramerNet, LinearRegression

import pprint
pp = pprint.PrettyPrinter(indent=4, width=160)

save_map = {
    'model': management.save_keras_model,
    'history': boiling_learning.utils.fold
    # 'history': management.save_pkl
}
load_map = {
    'model': management.load_keras_model,
    'history': boiling_learning.utils.fold
    # 'history': management.load_pkl
}

manager = management.ModelManager(
    file_name_fmt='{index}.model',
    save_method=management.save_serialized(save_map),
    load_method=management.load_serialized(load_map),
    printer=pp.pprint,
    verbose=1
)

optimizer_params = dict(
    lr=1e-4
)
reduce_lr_on_plateau_params = dict(
    monitor='val_loss', 
    factor=0.5, patience=5, min_delta=0.0001, min_lr=0, 
    verbose=0, mode='auto', cooldown=0
)

# kramer_net_params = dict(
#     input_shape=(244, 244, 3),
#     num_classes=3,
#     problem='classification',
#     compile_setup={
#         'do': False,
#         'params': dict(
#             optimizer=management.Value(
#                 {
#                     'name': 'adam',
#                     'params': optimizer_params
#                 }, 
#                 Adam(**optimizer_params)
#             ),
#             callback=management.Value(
#                 [
#                     {
#                         'name': 'ModelCheckpoint',
#                         'params': {
#                             'save_best_only': True
#                         }
#                     },
#                     {
#                         'name': 'ModelCheckpoint',
#                         'params': {
#                             'save_best_only': False
#                         }
#                     },
#                     {
#                         'name': 'EarlyStopping',
#                         'params': {
#                             'save_best_only': dict(
#                                 monitor='val_loss', 
#                                 min_delta=0, patience=10, baseline=None,
#                                 verbose=1, mode='auto',
#                                 restore_best_weights=True
#                             )
#                         }
#                     },
#                     {
#                         'name': 'ReduceLROnPlateau',
#                         'params': reduce_lr_on_plateau_params
#                     }
#                 ],
#                 [
#                     ModelCheckpoint(
#                         manager.models_path / 'BestValidation.h5',
#                         monitor='val_loss', verbose=1, 
#                         save_best_only=True
#                     ),
#                     ModelCheckpoint(
#                         manager.models_path / 'LastTrained.h5',
#                         monitor='val_loss', verbose=0,
#                         save_best_only=False
#                     ),
#                     EarlyStopping(
#                         monitor='val_loss', 
#                         min_delta=0, patience=10, baseline=None,
#                         verbose=1, mode='auto',
#                         restore_best_weights=True
#                     ),
#                     ReduceLROnPlateau(**reduce_lr_on_plateau_params)
#                 ]
#             )
#         )
#     },
#     fit_setup={
#         'do': False,
#         'method': 'fit_generator'
#     }
# )

# kramer_model = manager.provide_model(
#     KramerNet.creator,
#     params=kramer_net_params,
#     save=True,
#     load=True
# )
