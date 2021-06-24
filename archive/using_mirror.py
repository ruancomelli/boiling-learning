import pprint
from functools import partial
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import boiling_learning as bl
from boiling_learning.management import Mirror
from boiling_learning.model.definitions import HoboldNet2

pp = pprint.PrettyPrinter(indent=4, width=160)

n_train = n_val = 1000
IMG_SHAPE = [224, 224, 1]
BATCH_SIZE = 1028

grid_dict = {
    'lr': [1e-2, 1e-3, 1e-4],
    'reduce_lr_on_plateau_factor': [0.5, 0.2, 0.1],
    'reduce_lr_on_plateau_patience': [2, 4, 8],
    'dropout_ratio': [0.2, 0.5, 0.8],
}
grid = list(
    bl.utils.combine_dict(
        grid_dict,
        partial(bl.utils.alternate_iter, default_indices=[1, 1, 1, 1]),
    )
)
for params in grid:
    lr = params['lr']
    reduce_lr_on_plateau_factor = params['reduce_lr_on_plateau_factor']
    reduce_lr_on_plateau_patience = params['reduce_lr_on_plateau_patience']
    dropout_ratio = params['dropout_ratio']

    model_type = HoboldNet2
    optimizer_params = dict(
        lr=lr,
    )
    reduce_lr_on_plateau_params = dict(
        monitor='val_loss',
        factor=reduce_lr_on_plateau_factor,
        patience=reduce_lr_on_plateau_patience,
        min_delta=0.0001,
        min_lr=0,
        verbose=1,
        mode='auto',
        cooldown=0,
    )
    early_stopping_params = dict(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        baseline=None,
        verbose=1,
        mode='auto',
        restore_best_weights=True,
    )
    best_validation_callback_file_name = 'BestValidation_epoch{epoch}'
    best_validation_callback_params = dict(
        filepath=best_validation_callback_file_name,
        save_best_only=True,
        monitor='val_loss',
    )
    last_trained_callback_file_name = 'LastTrained_epoch{epoch}'
    last_trained_callback_params = dict(
        filepath=last_trained_callback_file_name,
        save_best_only=False,
        monitor='val_loss',
    )
    checkpoint_params = dict(
        restore=True,
        path=last_trained_callback_file_name,
    )

    model_params = Mirror(
        {'propagate': True},
        fetch=Mirror.Fork(value=['model', 'history']),
        verbose=Mirror.Fork(value=1),
        checkpoint=Mirror.Fork(
            desc=checkpoint_params,
            value=dict(checkpoint_params),
        ),
        input_shape=IMG_SHAPE,
        # num_classes=3,
        dropout_ratio=dropout_ratio,
        problem='regression',
        compile_setup=Mirror(
            {'propagate': True},
            do=True,
            params=Mirror(
                {'propagate': True},
                optimizer=Mirror.Fork(
                    desc={'name': 'adam', 'params': optimizer_params},
                    value=Adam(**optimizer_params),
                ),
                loss=Mirror.Fork(
                    desc='mean_squared_error',
                    value=tf.keras.losses.MeanSquaredError(),
                ),
                metrics=Mirror.Fork(
                    value=[
                        tf.keras.metrics.MeanSquaredError('MSE'),
                        tf.keras.metrics.RootMeanSquaredError('RMS'),
                        tf.keras.metrics.MeanAbsoluteError('MAE'),
                        tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
                        tfa.metrics.RSquare('R2'),
                    ]
                ),
            ),
        ),
        fit_setup=Mirror(
            {'propagate': True},
            do=True,
            params=Mirror(
                {'propagate': True},
                x=Mirror.Fork(
                    desc={
                        'source': Path('my_data') / 'path',
                        'train_size': n_train,
                    },
                ),
                validation_data=Mirror.Fork(
                    desc={
                        'source': Path('my_data') / 'path',
                        'val_size': n_val,
                    },
                ),
                verbose=Mirror.Fork(value=2),
                batch_size=BATCH_SIZE,
                epochs=100,
                use_multiprocessing=True,
                workers=-1,
                callbacks=Mirror.Fork(
                    desc=[
                        {
                            'name': 'ModelCheckpoint',
                            'params': best_validation_callback_params,
                        },
                        {
                            'name': 'ModelCheckpoint',
                            'params': last_trained_callback_params,
                        },
                        {
                            'name': 'EarlyStopping',
                            'params': early_stopping_params,
                        },
                        {
                            'name': 'ReduceLROnPlateau',
                            'params': reduce_lr_on_plateau_params,
                        },
                    ]
                ),
            ),
        ),
    )
    params_desc = model_params['desc']
    params_value = model_params['value']

    model_path = Path('my_model') / 'path'

    # This part is separated because ModelCheckpoint needs model_path
    best_validation_callback_path = str(
        model_path / best_validation_callback_file_name
    )
    best_validation_callback = ModelCheckpoint(
        **bl.utils.merge_dicts(
            best_validation_callback_params,
            dict(filepath=best_validation_callback_path, verbose=1),
        )
    )
    last_trained_callback_path = str(
        model_path / last_trained_callback_file_name
    )
    last_trained_callback = ModelCheckpoint(
        **bl.utils.merge_dicts(
            last_trained_callback_params,
            dict(filepath=last_trained_callback_path, verbose=1),
        )
    )
    early_stopping_callback = EarlyStopping(**early_stopping_params)
    reduce_lr_on_plateau_callback = ReduceLROnPlateau(
        **reduce_lr_on_plateau_params
    )

    params_value['fit_setup']['params']['callbacks'] = [
        best_validation_callback,
        last_trained_callback,
        early_stopping_callback,
        reduce_lr_on_plateau_callback,
    ]

    params_value['checkpoint']['path'] = last_trained_callback_path
    params_value['checkpoint']['load_method'] = load_model

    bl.utils.print_header('Description')
    pprint.pprint(params_desc)
    bl.utils.print_header('Value')
    pprint.pprint(params_value)

    break
