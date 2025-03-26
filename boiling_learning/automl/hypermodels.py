import typing
from collections.abc import Iterator
from typing import Any

import autokeras as ak
import keras_tuner as kt
import tensorflow as tf
from keras_tuner.engine.trial import Trial

from boiling_learning.automl.blocks import ImageNormalizationBlock, LayersBlock
from boiling_learning.automl.tuners import AutoTuner
from boiling_learning.io import json
from boiling_learning.lazy import Lazy
from boiling_learning.management.allocators import Allocator
from boiling_learning.model.model import ModelArchitecture, anonymize_model_json
from boiling_learning.utils.pathutils import PathLike, resolve


class HyperModel(kt.HyperModel):
    def __init__(self, automodel: ak.AutoModel) -> None:
        self.automodel = automodel

    @property
    def tuner(self) -> AutoTuner:
        return self.automodel.tuner

    def get_config(self) -> dict[str, Any]:
        return typing.cast(dict[str, Any], self.tuner.hypermodel.get_config())

    def __json_encode__(self) -> dict[str, Any]:
        return anonymize_model_json(
            {key: value for key, value in self.get_config().items() if key != "name"}
        )

    def __describe__(self) -> dict[str, Any]:
        return typing.cast(dict[str, Any], json.encode(self))

    def best_model(self) -> ModelArchitecture:
        return self.tuner.best_model()

    def best_trial(self) -> Trial:
        return self.tuner.best_trial()

    def best_hyperparameters(self) -> kt.HyperParameters:
        return self.tuner.best_hyperparameters()

    def iter_best_models(self) -> Iterator[ModelArchitecture]:
        return self.tuner.iter_best_models()

    def iter_scored_models(self) -> Iterator[tuple[ModelArchitecture, float]]:
        return self.tuner.iter_scored_models()


class ImageRegressor(HyperModel):
    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric],
        normalize_images: bool | None = None,
        augment_images: bool | None = None,
        directory: PathLike | Allocator | None = None,
        strategy: Lazy[tf.distribute.Strategy] | None = None,
        **kwargs: Any,
    ) -> None:
        if "overwrite" in kwargs:
            raise TypeError("the argument 'overwrite' is not supported.")

        if "distribution_strategy" in kwargs:
            raise TypeError("the argument 'distribution_strategy' is not supported.")

        inputs = ak.ImageInput()
        outputs = ak.ImageBlock(normalize=normalize_images, augment=augment_images)(
            inputs
        )
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
                distribution_strategy=strategy() if strategy is not None else None,
                **kwargs,
            )
        )


class ConvImageRegressor(HyperModel):
    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric],
        normalize_images: bool | None = None,
        directory: PathLike | Allocator | None = None,
        strategy: Lazy[tf.distribute.Strategy] | None = None,
        **kwargs: Any,
    ) -> None:
        if "overwrite" in kwargs:
            raise TypeError("the argument 'overwrite' is not supported.")

        if "distribution_strategy" in kwargs:
            raise TypeError("the argument 'distribution_strategy' is not supported.")

        outputs = inputs = ak.ImageInput()
        outputs = ImageNormalizationBlock(normalize_images)(outputs)
        outputs = ak.ConvBlock()(outputs)
        outputs = ak.SpatialReduction()(outputs)
        outputs = ak.DenseBlock()(outputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        if isinstance(directory, Allocator):
            directory = directory.allocate(
                self.__class__.__name__,
                loss=loss,
                metrics=metrics,
                normalize_images=normalize_images,
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
                distribution_strategy=strategy() if strategy is not None else None,
                **kwargs,
            )
        )


class FixedArchitectureImageRegressor(HyperModel):
    def __init__(
        self,
        layers: list[tf.keras.layers.Layer],
        loss: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric],
        directory: PathLike | Allocator | None = None,
        strategy: Lazy[tf.distribute.Strategy] | None = None,
        **kwargs: Any,
    ) -> None:
        if "overwrite" in kwargs:
            raise TypeError("the argument 'overwrite' is not supported.")

        if "distribution_strategy" in kwargs:
            raise TypeError("the argument 'distribution_strategy' is not supported.")

        layers_block = LayersBlock(layers)

        inputs = ak.ImageInput()
        outputs = layers_block(inputs)
        outputs = ak.RegressionHead(output_dim=1, loss=loss, metrics=metrics)(outputs)

        if isinstance(directory, Allocator):
            directory = directory.allocate(
                self.__class__.__name__,
                loss=loss,
                metrics=metrics,
                layers=layers_block.get_config(),
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
                distribution_strategy=strategy() if strategy is not None else None,
                **kwargs,
            )
        )
