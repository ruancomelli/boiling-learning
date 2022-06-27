import typing
from typing import Any, Dict, List, Optional

import autokeras as ak
import tensorflow as tf

from boiling_learning.automl.blocks import LayersBlock
from boiling_learning.io import json
from boiling_learning.model.model import anonymize_model_json


class HyperModel:
    def __init__(self, automodel: ak.AutoModel) -> None:
        self.automodel = automodel

    def get_config(self) -> Dict[str, Any]:
        return self.automodel.tuner.hypermodel.get_config()

    def __json_encode__(self) -> Dict[str, Any]:
        model_json = self.get_config()
        return anonymize_model_json(
            {key: value for key, value in model_json['config'].items() if key != 'name'}
        )

    def __describe__(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], json.encode(self))


class ImageRegressor(ak.AutoModel):
    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        metrics: List[tf.keras.metrics.Metric],
        normalize_images: Optional[bool] = None,
        augment_images: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        inputs = ak.ImageInput()
        outputs = ak.ImageBlock(normalize=normalize_images, augment=augment_images)(inputs)
        outputs = ak.SpatialReduction()(outputs)
        outputs = ak.DenseBlock()(outputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        super().__init__(inputs, outputs, **kwargs)


class FixedArchitectureImageRegressor(ak.AutoModel):
    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        loss: tf.keras.losses.Loss,
        metrics: List[tf.keras.metrics.Metric],
        **kwargs: Any,
    ) -> None:
        inputs = ak.ImageInput()
        outputs = LayersBlock(layers)(inputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        super().__init__(inputs, outputs, **kwargs)
