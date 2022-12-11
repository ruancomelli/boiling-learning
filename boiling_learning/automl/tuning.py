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
    hypermodel: HyperModel, datasets: DatasetTriplet[tf.data.Dataset], params: TuneModelParams
) -> TuneModelReturn:
    ds_train, ds_val, ds_test = datasets

    automodel = hypermodel.automodel
    automodel.fit(
        ds_train,
        validation_data=ds_val,
        callbacks=params.callbacks(),
        batch_size=params.batch_size,
    )

    model = ModelArchitecture(automodel.export_model())

    return TuneModelReturn(
        model=model,
        validation_metrics=model.evaluate(ds_val.batch(params.batch_size)),
        test_metrics=model.evaluate(ds_test.batch(params.batch_size)),
    )
