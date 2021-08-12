from fractions import Fraction
from typing import Dict, Optional, Union

import more_itertools as mit
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam

from boiling_learning.datasets import (
    DatasetSplits,
    apply_unbatched,
    calculate_dataset_size,
    train_val_test_split,
)
from boiling_learning.management import Manager, Parameters
from boiling_learning.model.callbacks import (
    AdditionalValidationSets,
    ReduceLROnPlateau,
    TimePrinter,
)
from boiling_learning.utils import ensure_dir, ensure_parent, merge_dicts
from boiling_learning.utils.functional import P, Pack


def main(
    manager: Manager,
    strategy: tf.distribute.Strategy,
    dataset_contents_train,
    dataset_contents_val,
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
    take_train: Optional[Union[int, Fraction]],
    take_val: Optional[Union[int, Fraction]],
    target: str,
    additional_val_sets: Dict[str, tf.data.Dataset],
    model_creator,
    lr,
    normalize_images: bool,
    reduce_lr_on_plateau_factor,
    reduce_lr_on_plateau_patience,
    early_stopping_patience,
    dropout_ratio,
    batch_size,
    missing_ok,
    include,
    hidden_layers_policy,
    output_layer_policy,
):
    def _take(ds, take):
        if take is None:
            return ds

        if isinstance(take, float):
            dataset_size = calculate_dataset_size(ds, batched_dim=0)
            take = int(take * dataset_size)

        if isinstance(take, int):
            return apply_unbatched(
                ds, lambda _ds: _ds.take(take), dim=0, key=0
            )
        elif isinstance(take, Fraction):
            return apply_unbatched(
                ds,
                lambda _ds: train_val_test_split(
                    _ds, splits=DatasetSplits(train=take), shuffle=False
                )[0],
                dim=0,
                key=0,
            )

    ds_train = _take(ds_train, take_train)
    ds_val = _take(ds_val, take_val)

    def _get_target_pair(img, data):
        return img, data[target]

    ds_train = ds_train.map(_get_target_pair)
    ds_val = ds_val.map(_get_target_pair)

    additional_val_sets = tuple(
        (additional_val_set.map(_get_target_pair), name)
        for name, additional_val_set in additional_val_sets.items()
    )

    sample_image = mit.first(ds_train.unbatch().take(1).as_numpy_iterator())[0]
    img_shape = sample_image.shape

    optimizer_params = dict(lr=lr)

    additional_validation_sets = AdditionalValidationSets(
        additional_val_sets, verbose=1, batch_size=batch_size
    )

    use_reduce_lr_on_plateau = None not in {
        reduce_lr_on_plateau_factor,
        reduce_lr_on_plateau_patience,
    }
    if use_reduce_lr_on_plateau:
        reduce_lr_on_plateau_params = dict(
            monitor='val_loss',
            factor=reduce_lr_on_plateau_factor,
            patience=reduce_lr_on_plateau_patience,
            min_delta=0.01,
            min_delta_mode='relative',
            min_lr=0,
            mode='auto',
            cooldown=2,
        )
    early_stopping_params = dict(
        monitor='val_loss',
        min_delta=0,
        patience=early_stopping_patience,
        baseline=None,
        mode='auto',
        restore_best_weights=True,
    )
    last_trained_callback_file_name = 'last_trained'
    last_trained_callback_params = dict(
        filepath=last_trained_callback_file_name,
        save_best_only=False,
        monitor='val_loss',
    )
    backup_dir_name = 'backup_dir'

    with strategy.scope():
        metrics = [
            tf.keras.metrics.MeanSquaredError('MSE'),
            tf.keras.metrics.RootMeanSquaredError('RMS'),
            tf.keras.metrics.MeanAbsoluteError('MAE'),
            tf.keras.metrics.MeanAbsolutePercentageError('MAPE'),
            # tfa.metrics.RSquare('R2')
            # RSquare('R2', y_shape=(1,))
        ]

    model_params = Parameters()
    model_params[['value', 'num_classes']] = None
    # model_params[['desc', 'strategy']] = strategy_name
    model_params[['value', 'strategy']] = strategy
    model_params[['value', 'fetch']] = ['model', 'history']
    # model_params[['value', 'verbose']] = 1
    model_params[[{'desc', 'value'}, 'problem']] = 'regression'
    model_params[
        [{'desc', 'value'}, 'architecture_setup', 'input_shape']
    ] = img_shape
    model_params[
        [{'desc', 'value'}, 'architecture_setup', 'dropout']
    ] = dropout_ratio
    if normalize_images:
        model_params[
            [{'desc', 'value'}, 'architecture_setup', 'normalize_images']
        ] = normalize_images
    model_params[
        ['desc', 'architecture_setup', 'hidden_layers_policy']
    ] = hidden_layers_policy
    model_params[
        ['value', 'architecture_setup', 'hidden_layers_policy']
    ] = mixed_precision.Policy(hidden_layers_policy)
    model_params[
        ['desc', 'architecture_setup', 'output_layer_policy']
    ] = output_layer_policy
    model_params[
        ['value', 'architecture_setup', 'output_layer_policy']
    ] = mixed_precision.Policy(output_layer_policy)
    model_params[[{'desc', 'value'}, 'compile_setup', 'do']] = True
    model_params[['desc', 'compile_setup', 'params', 'optimizer']] = {
        'name': 'Adam',
        'params': optimizer_params,
    }
    model_params[['value', 'compile_setup', 'params', 'optimizer']] = Adam(
        **optimizer_params
    )
    model_params[
        ['desc', 'compile_setup', 'params', 'loss']
    ] = 'mean_squared_error'
    model_params[
        ['value', 'compile_setup', 'params', 'loss']
    ] = tf.keras.losses.MeanSquaredError()
    model_params[['desc', 'compile_setup', 'params', 'metrics']] = [
        metric.name for metric in metrics
    ]
    model_params[['value', 'compile_setup', 'params', 'metrics']] = metrics
    model_params[[{'desc', 'value'}, 'fit_setup', 'do']] = True
    model_params[
        ['desc', 'fit_setup', 'params', 'x', 'dataset_contents']
    ] = dataset_contents_train
    model_params[['desc', 'fit_setup', 'params', 'x', 'take']] = take_train
    model_params[['value', 'fit_setup', 'params', 'x']] = ds_train
    model_params[
        ['desc', 'fit_setup', 'params', 'validation_data', 'dataset_contents']
    ] = dataset_contents_val
    model_params[
        ['desc', 'fit_setup', 'params', 'validation_data', 'take']
    ] = take_val
    model_params[['value', 'fit_setup', 'params', 'validation_data']] = ds_val
    model_params[['value', 'fit_setup', 'params', 'verbose']] = 2
    model_params[
        [{'desc', 'value'}, 'fit_setup', 'params', 'batch_size']
    ] = batch_size
    model_params[[{'desc', 'value'}, 'fit_setup', 'params', 'epochs']] = 100
    model_params[
        [{'desc', 'value'}, 'fit_setup', 'params', 'use_multiprocessing']
    ] = True
    model_params[[{'desc', 'value'}, 'fit_setup', 'params', 'workers']] = -1
    model_params[['desc', 'fit_setup', 'params', 'callbacks']] = [
        {
            'name': 'BackupAndRestore',
            'params': {'backup_dir_name': backup_dir_name},
        },
        {'name': 'ModelCheckpoint', 'params': last_trained_callback_params},
        {'name': 'EarlyStopping', 'params': early_stopping_params},
        {'name': 'TerminateOnNaN', 'params': P()},
    ]
    if use_reduce_lr_on_plateau:
        model_params[['desc', 'fit_setup', 'params', 'callbacks']].append(
            {
                'name': 'ReduceLROnPlateau',
                'params': reduce_lr_on_plateau_params,
            }
        )

    model_id = manager.provide_entry(
        creator=model_creator,
        creator_description=Pack(kwargs=model_params['desc']),
        post_processor=None,
        post_processor_description=P(),
        missing_ok=missing_ok,
        include=include,
    )

    # This part is separated because ModelCheckpoint needs model_workspace
    model_workspace = manager.elem_workspace(model_id)
    checkpoints_dir = ensure_dir(model_workspace / 'checkpoints')
    backup_dir = ensure_parent(model_workspace / backup_dir_name)

    last_trained_callback_path = (
        checkpoints_dir / last_trained_callback_file_name
    )
    last_trained_callback = ModelCheckpoint(
        **merge_dicts(
            last_trained_callback_params,
            dict(filepath=str(last_trained_callback_path), verbose=1),
        )
    )

    early_stopping_callback = EarlyStopping(
        **merge_dicts(early_stopping_params, dict(verbose=1))
    )

    model_params.setdefault(
        ['value', 'fit_setup', 'params', 'callbacks'], []
    ).extend(
        [
            last_trained_callback,
            early_stopping_callback,
            additional_validation_sets,
            TimePrinter(),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.experimental.BackupAndRestore(str(backup_dir)),
        ]
    )
    if use_reduce_lr_on_plateau:
        model_params[['value', 'fit_setup', 'params', 'callbacks']].append(
            ReduceLROnPlateau(
                **merge_dicts(reduce_lr_on_plateau_params, dict(verbose=1))
            )
        )

    return model_id, manager.provide_elem(
        elem_id=model_id,
        creator=model_creator,
        creator_params=Pack(kwargs=model_params['value']),
        post_processor=None,
        save=True,
        load=True,
    )


if __name__ == '__main__':
    raise RuntimeError(
        '*make_model* cannot be executed as a standalone script yet.'
    )
