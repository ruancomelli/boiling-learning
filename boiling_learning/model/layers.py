from typing import Any

import numpy as np
import tensorflow as tf


class RandomBrightness(tf.keras.layers.Layer):
    """Backport of TF 2.9 `tf.keras.layers.RandomBrightness` to TF 2.8.

    Implementation based on
    https://towardsdatascience.com/writing-a-custom-data-augmentation-layer-in-keras-2b53e048a98
    """

    def __init__(self, factor: float | tuple[float, float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._factor = (
            (-abs(factor), abs(factor))
            if isinstance(factor, float)
            else (min(factor), max(factor))
        )

    def call(self, images: Any, training: bool = False) -> Any:
        if not training:
            return images

        min_brightness, max_brightness = self._factor
        brightness = np.random.uniform(min_brightness, max_brightness)

        return tf.image.adjust_brightness(images, brightness)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "factor": self._factor}


class ImageNormalization(tf.keras.layers.Layer):
    def call(self, images: Any, training: bool = False) -> Any:
        return tf.image.per_image_standardization(images)
