from typing import (
    Any,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union
)

import numpy as np
from skimage.transform import AffineTransform, warp
import tensorflow as tf

T = TypeVar('T')
ImageType = Any # something convertible to tf.Tensor


def _ratio_to_size(image: ImageType, x: Optional[Union[int, float]], axis: int) -> Optional[int]:
    if isinstance(x, float):
        return int(x * image.shape[axis])
    else:
        return x


def crop(
        image: ImageType,
        left: Optional[Union[int, float]] = None,
        right: Optional[Union[int, float]] = None,
        top: Optional[Union[int, float]] = None,
        bottom: Optional[Union[int, float]] = None,
        height: Optional[Union[int, float]] = None,
        width: Optional[Union[int, float]] = None
) -> tf.Tensor:
    if (left, right, width).count(None) != 1:
        raise ValueError('exactly one of *left*, *right* and *width* must be None')
    if (top, bottom, height).count(None) != 1:
        raise ValueError('exactly one of *top*, *bottom* and *height* must be None')

    left = _ratio_to_size(image, left, axis=1)
    right = _ratio_to_size(image, right, axis=1)
    width = _ratio_to_size(image, width, axis=1)
    top = _ratio_to_size(image, top, axis=0)
    bottom = _ratio_to_size(image, bottom, axis=0)
    height = _ratio_to_size(image, height, axis=0)

    if top is None:
        top = bottom - height
    if height is None:
        height = bottom - top

    if left is None:
        left = right - width
    if width is None:
        width = right - left

    return tf.image.crop_to_bounding_box(
        image,
        offset_height=top,
        offset_width=left,
        target_height=height,
        target_width=width
    )


def shrink(
        image: ImageType,
        left: Optional[Union[int, float]] = None,
        right: Optional[Union[int, float]] = None,
        top: Optional[Union[int, float]] = None,
        bottom: Optional[Union[int, float]] = None,
        height: Optional[Union[int, float]] = None,
        width: Optional[Union[int, float]] = None
) -> tf.Tensor:
    if (left, right, width).count(None) != 1:
        raise ValueError('exactly one of *left*, *right* and *width* must be None')
    if (top, bottom, height).count(None) != 1:
        raise ValueError('exactly one of *top*, *bottom* and *height* must be None')

    left = _ratio_to_size(image, left, axis=1)
    right = _ratio_to_size(image, right, axis=1)
    width = _ratio_to_size(image, width, axis=1)
    top = _ratio_to_size(image, top, axis=0)
    bottom = _ratio_to_size(image, bottom, axis=0)
    height = _ratio_to_size(image, height, axis=0)

    if top is None:
        top = image.shape[0] - (bottom + height)
    if height is None:
        height = image.shape[0] - (bottom + top)

    if left is None:
        left = image.shape[1] - (right + width)
    if width is None:
        width = image.shape[1] - (right + left)

    return tf.image.crop_to_bounding_box(
        image,
        offset_height=top,
        offset_width=left,
        target_height=height,
        target_width=width
    )


def shift(
        image: np.ndarray,
        shift_left: Optional[Union[int, float]] = None,
        shift_right: Optional[Union[int, float]] = None,
        shift_up: Optional[Union[int, float]] = None,
        shift_down: Optional[Union[int, float]] = None
) -> np.ndarray:
    # source: <https://stackoverflow.com/questions/47961447/shift-image-in-scikit-image-python>

    if (shift_left, shift_right).count(None) != 1:
        raise ValueError('exactly one of *shift_left* and *shift_right* must be None')
    if (shift_down, shift_up).count(None) != 1:
        raise ValueError('exactly one of *shift_down* and *shift_up* must be None')

    shift_left = _ratio_to_size(image, shift_left, axis=1)
    shift_right = _ratio_to_size(image, shift_right, axis=1)
    shift_up = _ratio_to_size(image, shift_up, axis=0)
    shift_down = _ratio_to_size(image, shift_down, axis=0)

    if shift_left is None:
        shift_left = -shift_right
    if shift_up is None:
        shift_up = -shift_down
    shifts = (shift_left, shift_up)

    transform = AffineTransform(translation=shifts)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)

    return shifted


def flip(
        image: T,
        horizontal: bool = False,
        vertical: bool = False
) -> Union[T, tf.Tensor]:
    if horizontal:
        image = tf.image.flip_left_right(image)
    if vertical:
        image = tf.image.flip_up_down(image)

    return image


def grayscale(image: ImageType) -> tf.Tensor:
    return tf.image.rgb_to_grayscale(image)


def downscale(
        image: ImageType,
        factors: Tuple[int, int],
        antialias: bool = False
) -> tf.Tensor:
    sizes = (image.shape[0]//factors[0], image.shape[1]//factors[1])
    return tf.image.resize(image, sizes, method='bilinear', antialias=antialias)


def random_brightness(image: ImageType, min_delta: float, max_delta: float) -> tf.Tensor:
    delta = tf.random.uniform([], minval=min_delta, maxval=max_delta)
    return tf.image.adjust_brightness(image, delta)


def random_crop(
        image: ImageType,
        size: Iterable[Optional[int]],
        seed=None
) -> tf.Tensor:
    size = tuple(
        dim if dim is not None else img_dim
        for img_dim, dim in zip(image.shape, size)
    )
    return tf.image.random_crop(image, size, seed=seed)
