from collections.abc import Callable

import tensorflow as tf

from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.training.common import cached_fit_model_function
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.callbacks import MemoryCleanUp, RegisterEpoch, SaveHistory, TimePrinter
from boiling_learning.model.training import CompiledModel, FitModelParams, FitModelReturn
from boiling_learning.utils.functional import P
from boiling_learning.utils.pathutils import resolve


def fit_condensation_model(
    compiled_model: CompiledModel,
    datasets: LazyDescribed[ImageDatasetTriplet],
    params: FitModelParams,
    *,
    strategy: LazyDescribed[tf.distribute.Strategy],
    target: str,
) -> FitModelReturn:
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
            MemoryCleanUp(),
        )
    )

    fitter = cached_fit_model_function('condensation', strategy=strategy)

    workspace_path = resolve(
        fitter.allocate(compiled_model, datasets, params, target),
        parents=True,
    )

    creator: Callable[[], FitModelReturn] = P(
        compiled_model,
        tuple(
            subset()
            for subset in to_tensorflow_triplet(
                datasets,
                batch_size=params.batch_size,
                include_test=False,
                target=target,
                experiment='condensation',
            )
        ),
        params,
        epoch_registry=RegisterEpoch(workspace_path / 'epoch.json'),
        history_registry=SaveHistory(workspace_path / 'history.json', mode='a'),
    ).partial(fitter.function)

    return fitter.provide(creator, workspace_path / 'model')
