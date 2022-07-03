import typing
from typing import Any, Dict, List, Optional, Union

import autokeras as ak
import keras_tuner as kt
import tensorflow as tf

from boiling_learning.automl.blocks import LayersBlock
from boiling_learning.io import json
from boiling_learning.management.allocators import Allocator
from boiling_learning.model.model import anonymize_model_json
from boiling_learning.utils.described import Described
from boiling_learning.utils.utils import PathLike, resolve


class HyperModel(kt.HyperModel):
    def __init__(self, automodel: ak.AutoModel) -> None:
        self.automodel = automodel

    def get_config(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], self.automodel.tuner.hypermodel.get_config())

    def __json_encode__(self) -> Dict[str, Any]:
        return anonymize_model_json(
            {key: value for key, value in self.get_config().items() if key != 'name'}
        )

    def __describe__(self) -> Dict[str, Any]:
        return typing.cast(Dict[str, Any], json.encode(self))


class ImageRegressor(HyperModel):
    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        metrics: List[tf.keras.metrics.Metric],
        normalize_images: Optional[bool] = None,
        augment_images: Optional[bool] = None,
        directory: Union[PathLike, Allocator, None] = None,
        strategy: Optional[Described[tf.distribute.Strategy, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if 'overwrite' in kwargs:
            raise TypeError("the argument 'overwrite' is not supported.")

        if 'distribution_strategy' in kwargs:
            raise TypeError("the argument 'distribution_strategy' is not supported.")

        inputs = ak.ImageInput()
        outputs = ak.ImageBlock(normalize=normalize_images, augment=augment_images)(inputs)
        outputs = ak.SpatialReduction()(outputs)
        outputs = ak.DenseBlock()(outputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        if isinstance(directory, Allocator):
            directory = directory.allocate(
                self.__class__.__name__,
                loss=loss,
                metrics=metrics,
                normalize_images=normalize_images,
                augment_images=augment_images,
                **kwargs,
            )
        elif directory is not None:
            directory = resolve(directory, parents=True)

        super().__init__(
            ak.AutoModel(
                inputs,
                outputs,
                directory=directory,
                overwrite=directory is None,
                distribution_strategy=strategy.value if strategy is not None else None,
                **kwargs,
            )
        )


class FixedArchitectureImageRegressor(HyperModel):
    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        loss: tf.keras.losses.Loss,
        metrics: List[tf.keras.metrics.Metric],
        directory: Union[PathLike, Allocator, None] = None,
        strategy: Optional[Described[tf.distribute.Strategy, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if 'overwrite' in kwargs:
            raise TypeError("the argument 'overwrite' is not supported.")

        if 'distribution_strategy' in kwargs:
            raise TypeError("the argument 'distribution_strategy' is not supported.")

        inputs = ak.ImageInput()
        outputs = LayersBlock(layers)(inputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        if isinstance(directory, Allocator):
            directory = directory.allocate(
                self.__class__.__name__,
                loss=loss,
                metrics=metrics,
                layers=[layer.get_config() for layer in layers],
                **kwargs,
            )
        elif directory is not None:
            directory = resolve(directory, parents=True)

        super().__init__(
            ak.AutoModel(
                inputs,
                outputs,
                directory=directory,
                overwrite=directory is None,
                distribution_strategy=strategy.value if strategy is not None else None,
                **kwargs,
            )
        )
