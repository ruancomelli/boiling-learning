import tensorflow as tf

from boiling_learning.automl.hypermodels import HyperModel
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.io.storage import dataclass
from boiling_learning.lazy import LazyDescribed
from boiling_learning.model.model import Evaluation, ModelArchitecture


@dataclass(frozen=True)
class TuneModelParams:
    callbacks: LazyDescribed[list[tf.keras.callbacks.Callback]]
    batch_size: int


@dataclass(frozen=True)
class TuneModelReturn:
    model: ModelArchitecture
    validation_metrics: Evaluation
    test_metrics: Evaluation


def fit_hypermodel(
    hypermodel: HyperModel,
    datasets: DatasetTriplet[LazyDescribed[tf.data.Dataset]],
    params: TuneModelParams,
) -> TuneModelReturn:
    lazy_ds_train, lazy_ds_val, lazy_ds_test = datasets
    ds_train = lazy_ds_train().prefetch(tf.data.AUTOTUNE)
    ds_val = lazy_ds_val().prefetch(tf.data.AUTOTUNE)
    ds_test = lazy_ds_test().prefetch(tf.data.AUTOTUNE)

    automodel = hypermodel.automodel
    automodel.fit(
        ds_train,
        validation_data=ds_val,
        callbacks=params.callbacks(),
        batch_size=params.batch_size,
    )

    model = hypermodel.best_model()

    return TuneModelReturn(
        model=model,
        validation_metrics=model.evaluate(ds_val.batch(params.batch_size)),
        test_metrics=model.evaluate(ds_test.batch(params.batch_size)),
    )
