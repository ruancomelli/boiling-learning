from collections.abc import Callable

import tensorflow as tf

from boiling_learning.app.datasets.bridged.boiling1d import (
    DEFAULT_BOILING_HEAT_FLUX_TARGET,
    default_boiling_bridging,
    default_boiling_bridging_gt10,
)
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.common import (
    LazyFitModelReturn,
    cached_fit_model_function,
    get_baseline_architecture,
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.callbacks import (
    AdditionalValidationSets,
    MemoryCleanUp,
    RegisterEpoch,
    SaveHistory,
    TimePrinter,
)
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import FitModelParams, FitModelReturn, compile_model
from boiling_learning.utils.functional import P
from boiling_learning.utils.pathutils import resolve


def fit_boiling_model(
    model: LazyDescribed[ModelArchitecture],
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: FitModelParams,
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    try_id: int = 0,
    target: str = DEFAULT_BOILING_HEAT_FLUX_TARGET,
) -> LazyFitModelReturn:
    """
    try_id: use this to force this model to be trained again.
        This may be used for instance to get a average and stddev.
    """

    _, ds_val_g10, _ = default_boiling_bridging_gt10(
        datasets,
        batch_size=params.batch_size,
        target=target,
    )

    params.callbacks().extend(
        (
            TimePrinter(
                when={
                    'on_epoch_begin',
                    'on_epoch_end',
                    'on_predict_begin',
                    'on_predict_end',
                    'on_test_begin',
                    'on_test_end',
                    'on_train_begin',
                    'on_train_end',
                }
            ),
            # BackupAndRestore(workspace_path / 'backup', delete_on_end=False),
            AdditionalValidationSets({'HF10': ds_val_g10()}),
            MemoryCleanUp(),
        )
    )

    fitter = cached_fit_model_function('boiling1d', strategy=strategy)

    workspace_path = resolve(
        fitter.allocate(model, datasets, params, target, try_id),
        parents=True,
    )

    creator: Callable[[], FitModelReturn] = P(
        model(),
        tuple(
            subset()
            for subset in default_boiling_bridging(
                datasets,
                batch_size=params.batch_size,
                target=target,
            )
        ),
        params,
        epoch_registry=RegisterEpoch(workspace_path / 'epoch.json'),
        history_registry=SaveHistory(workspace_path / 'history.json', mode='a'),
    ).partial(fitter.function)

    fit_model_return = fitter.provide(creator, workspace_path / 'model')

    return LazyFitModelReturn(
        LazyDescribed.from_value_and_description(
            fit_model_return.architecture,
            (model, datasets, params, target, try_id),
        ),
        trained_epochs=fit_model_return.trained_epochs,
        history=fit_model_return.history,
        train_time=fit_model_return.train_time,
        validation_metrics=fit_model_return.validation_metrics,
        test_metrics=fit_model_return.test_metrics,
    )


def get_baseline_boiling_architecture(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    direct_visualization: bool = True,
    normalize_images: bool = True,
) -> LazyDescribed[ModelArchitecture]:
    return get_baseline_architecture(
        baseline_boiling_dataset(direct_visualization=direct_visualization),
        strategy=strategy,
        normalize_images=normalize_images,
    )


def get_pretrained_baseline_boiling_model(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    direct_visualization: bool = True,
    normalize_images: bool = True,
) -> LazyFitModelReturn:
    model = get_baseline_boiling_architecture(
        direct_visualization=direct_visualization,
        normalize_images=normalize_images,
        strategy=strategy,
    ) | compile_model(
        **get_baseline_compile_params(strategy=strategy),
    )

    return fit_boiling_model(
        model,
        baseline_boiling_dataset(direct_visualization=direct_visualization),
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
    )
