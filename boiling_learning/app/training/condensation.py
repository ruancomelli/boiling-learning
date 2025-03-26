from collections.abc import Callable

import tensorflow as tf

from boiling_learning.app.constants import DEFAULT_CONDENSATION_MASS_RATE_TARGET
from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.datasets.preprocessed.condensation import condensation_dataset
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
    MemoryCleanUp,
    RegisterEpoch,
    SaveHistory,
    TimePrinter,
)
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.model.training import (
    FitModelParams,
    FitModelReturn,
    compile_model,
)
from boiling_learning.utils.functional import P
from boiling_learning.utils.pathutils import resolve


def fit_condensation_model(
    model: LazyDescribed[ModelArchitecture],
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: FitModelParams,
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
) -> LazyFitModelReturn:
    params.callbacks().extend(
        (
            TimePrinter(
                when={
                    "on_epoch_begin",
                    "on_epoch_end",
                    "on_predict_begin",
                    "on_predict_end",
                    "on_test_begin",
                    "on_test_end",
                    "on_train_begin",
                    "on_train_end",
                }
            ),
            # BackupAndRestore(workspace_path / 'backup', delete_on_end=False),
            MemoryCleanUp(),
        )
    )

    fitter = cached_fit_model_function("condensation", strategy=strategy)

    workspace_path = resolve(
        fitter.allocate(model, datasets, params, target),
        parents=True,
    )

    creator: Callable[[], FitModelReturn] = P(
        model(),
        tuple(
            subset()
            for subset in to_tensorflow_triplet(
                datasets,
                batch_size=params.batch_size,
                target=target,
                experiment="condensation",
            )
        ),
        params,
        epoch_registry=RegisterEpoch(workspace_path / "epoch.json"),
        history_registry=SaveHistory(workspace_path / "history.json", mode="a"),
    ).partial(fitter.function)

    fit_model_return = fitter.provide(creator, workspace_path / "model")

    return LazyFitModelReturn(
        LazyDescribed.from_value_and_description(
            fit_model_return.architecture,
            (
                model,
                datasets,
                params,
                target,
            ),
        ),
        trained_epochs=fit_model_return.trained_epochs,
        history=fit_model_return.history,
        train_time=fit_model_return.train_time,
        validation_metrics=fit_model_return.validation_metrics,
        test_metrics=fit_model_return.test_metrics,
    )


def get_baseline_condensation_architecture(
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    normalize_images: bool = True,
) -> LazyDescribed[ModelArchitecture]:
    return get_baseline_architecture(
        condensation_dataset(each=1),
        strategy=strategy,
        normalize_images=normalize_images,
    )


def get_pretrained_baseline_condensation_model(
    *,
    each: int,
    strategy: LazyDescribed[tf.distribute.Strategy],
    normalize_images: bool = True,
    target: str = DEFAULT_CONDENSATION_MASS_RATE_TARGET,
) -> LazyFitModelReturn:
    compiled_model = get_baseline_condensation_architecture(
        normalize_images=normalize_images,
        strategy=strategy,
    ) | compile_model(
        **get_baseline_compile_params(strategy=strategy),
    )

    return fit_condensation_model(
        compiled_model,
        condensation_dataset(each=each),
        get_baseline_fit_params(),
        target=target,
        strategy=strategy,
    )
