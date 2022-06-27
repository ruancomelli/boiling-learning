from typing import List, Optional

import tensorflow as tf

from boiling_learning.automl.hypermodels import HyperModel
from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.io import json
from boiling_learning.model.model import ModelArchitecture
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.described import Described


@dataclass(frozen=True)
class TuneModelParams:
    callbacks: Described[List[tf.keras.callbacks.Callback], json.JSONDataType]
    batch_size: Optional[int] = None


def fit_hypermodel(
    hypermodel: HyperModel, datasets: DatasetTriplet[tf.data.Dataset], params: TuneModelParams
) -> ModelArchitecture:
    ds_train, ds_val, _ = datasets

    automodel = hypermodel.automodel
    automodel.fit(
        ds_train,
        validation_data=ds_val,
        callbacks=params.callbacks,
        batch_size=params.batch_size,
    )

    return ModelArchitecture(automodel.export_model())
