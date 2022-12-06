from collections.abc import Callable

import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow, to_tensorflow_triplet
from boiling_learning.app.datasets.preprocessed.boiling1d import baseline_boiling_dataset
from boiling_learning.app.training.common import (
    cached_fit_model_function,
    get_baseline_architecture,
    get_baseline_compile_params,
    get_baseline_fit_params,
)
from boiling_learning.image_datasets import Image, ImageDatasetTriplet, Targets
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.callbacks import (
    AdditionalValidationSets,
    MemoryCleanUp,
    RegisterEpoch,
    SaveHistory,
    TimePrinter,
)
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import (
    CompiledModel,
    FitModelParams,
    FitModelReturn,
    compile_model,
)
from boiling_learning.transforms import subset
from boiling_learning.utils.functional import P
from boiling_learning.utils.pathutils import resolve


def _boiling_outlier_filter(_image: Image, target: Targets) -> bool:
    return abs(target['Power [W]'] - target['nominal_power']) < 5


DEFAULT_BOILING_OUTLIER_FILTER = LazyDescribed.from_value_and_description(
    _boiling_outlier_filter, 'abs(Power [W] - nominal_power) < 5'
)
DEFAULT_BOILING_HEAT_FLUX_TARGET = 'Flux [W/cm**2]'


def fit_boiling_model(
    compiled_model: CompiledModel,
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: FitModelParams,
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    try_id: int = 0,
    target: str = DEFAULT_BOILING_HEAT_FLUX_TARGET,
) -> FitModelReturn:
    """
    try_id: use this to force this model to be trained again.
        This may be used for instance to get a average and stddev.
    """

    def _is_gt10(_frame: Image, data: Targets) -> bool:
        return data[target] >= 10

    ds_val_g10 = to_tensorflow(
        datasets | subset('val'),
        prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
        filterer=_is_gt10,
        batch_size=params.batch_size,
        target=target,
        experiment='boiling1d',
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
        fitter.allocate(compiled_model, datasets, params, target, try_id),
        parents=True,
    )

    creator: Callable[[], FitModelReturn] = P(
        compiled_model,
        tuple(
            subset()
            for subset in to_tensorflow_triplet(
                datasets,
                prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
                batch_size=params.batch_size,
                target=target,
                experiment='boiling1d',
            )
        ),
        params,
        epoch_registry=RegisterEpoch(workspace_path / 'epoch.json'),
        history_registry=SaveHistory(workspace_path / 'history.json', mode='a'),
    ).partial(fitter.function)

    return fitter.provide(creator, workspace_path / 'model')


def get_baseline_boiling_architecture(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    direct_visualization: bool = True,
    normalize_images: bool = True,
) -> ModelArchitecture:
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
) -> FitModelReturn:
    compiled_model = compile_model(
        get_baseline_boiling_architecture(
            direct_visualization=direct_visualization,
            normalize_images=normalize_images,
            strategy=strategy,
        ),
        get_baseline_compile_params(strategy=strategy),
    )

    return fit_boiling_model(
        compiled_model,
        baseline_boiling_dataset(direct_visualization=direct_visualization),
        get_baseline_fit_params(),
        target=DEFAULT_BOILING_HEAT_FLUX_TARGET,
        strategy=strategy,
    )
