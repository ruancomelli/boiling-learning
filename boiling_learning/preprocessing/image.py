import typing
from fractions import Fraction
from typing import Optional, TypeVar, Union, overload

import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from skimage.measure import shannon_entropy
from skimage.metrics import normalized_mutual_information as _normalized_mutual_information
from skimage.metrics import structural_similarity as _structural_similarity
from skimage.transform import resize

from boiling_learning.preprocessing.transformers import wrap_as_partial_transformer
from boiling_learning.preprocessing.video import VideoFrame, VideoFrames

VideoFrameOrFrames = Union[VideoFrame, VideoFrames]
_VideoFrameOrFrames = TypeVar('_VideoFrameOrFrames', bound=VideoFrameOrFrames)


@wrap_as_partial_transformer
def image_dtype_converter(image: _VideoFrameOrFrames, dtype: str) -> _VideoFrameOrFrames:
    return tf.image.convert_image_dtype(image, tf.dtypes.as_dtype(dtype)).numpy()


@wrap_as_partial_transformer
def cropper(
    image: _VideoFrameOrFrames,
    *,
    left: Optional[Union[int, float, Fraction]] = None,
    right: Optional[Union[int, float, Fraction]] = None,
    right_border: Optional[Union[int, float, Fraction]] = None,
    width: Optional[Union[int, float, Fraction]] = None,
    top: Optional[Union[int, float, Fraction]] = None,
    bottom: Optional[Union[int, float, Fraction]] = None,
    bottom_border: Optional[Union[int, float, Fraction]] = None,
    height: Optional[Union[int, float, Fraction]] = None,
) -> _VideoFrameOrFrames:
    if image.ndim == 3:
        total_height, total_width, _ = image.shape
    elif image.ndim == 4:
        _, total_height, total_width, _ = image.shape
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    left = _ratio_to_size(total_width, left)
    right = _ratio_to_size(total_width, right)
    right_border = _ratio_to_size(total_width, right_border)
    width = _ratio_to_size(total_width, width)
    top = _ratio_to_size(total_height, top)
    bottom = _ratio_to_size(total_height, bottom)
    bottom_border = _ratio_to_size(total_height, bottom_border)
    height = _ratio_to_size(total_height, height)

    incompatible_arguments_error_message = (
        'at least two of `{}`, `{}`, `{}` and `{}` must be `None` or omitted.'
    )
    incompatible_pair_error_message = 'at least one of `{}` and `{}` must be `None` or omitted.'

    if (left, right, right_border, width).count(None) < 2:
        raise TypeError(
            incompatible_arguments_error_message.format('left', 'right', 'right_border', 'width')
        )

    if (top, bottom, bottom_border, height).count(None) < 2:
        raise TypeError(
            incompatible_arguments_error_message.format('top', 'bottom', 'bottom_border', 'height')
        )

    if right_border is not None:
        if right is not None:
            raise TypeError(incompatible_pair_error_message.format('right', 'right_border'))
        right = total_width - right_border

    if bottom_border is not None:
        if bottom is not None:
            raise TypeError(incompatible_pair_error_message.format('bottom', 'bottom_border'))
        bottom = total_height - bottom_border

    if width is not None:
        if right is not None:
            left = right - width
        elif left is not None:
            right = left + width

    if height is not None:
        if bottom is not None:
            top = bottom - height
        elif top is not None:
            bottom = top + height

    left = left if left is not None else 0
    right = right if right is not None else total_width
    top = top if top is not None else 0
    bottom = bottom if bottom is not None else total_height
    height = height if height is not None else bottom - top
    width = width if width is not None else right - left

    assert 0 <= left <= right
    assert 0 <= right <= total_width
    assert 0 <= top <= bottom
    assert 0 <= bottom <= total_height

    return typing.cast(
        _VideoFrameOrFrames,
        image[
            ...,  # support a batch axis
            top:bottom,  # crop vertically
            left:right,  # crop horizontally
            :,  # don't crop the channel axis
        ],
    )


@wrap_as_partial_transformer
def center_cropper(
    image: _VideoFrameOrFrames,
    *,
    height: Optional[Union[int, float, Fraction]] = None,
    width: Optional[Union[int, float, Fraction]] = None,
) -> _VideoFrameOrFrames:
    if image.ndim == 3:
        total_height, total_width, _ = image.shape
    elif image.ndim == 4:
        _, total_height, total_width, _ = image.shape
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    height = _ratio_to_size(total_width, height) if height is not None else total_height
    width = _ratio_to_size(total_width, width) if width is not None else total_width

    return typing.cast(
        _VideoFrameOrFrames, tf.keras.layers.CenterCrop(height, width)(image).numpy()
    )


@overload
def _ratio_to_size(total: int, x: Union[int, float, Fraction]) -> int:
    ...


@overload
def _ratio_to_size(total: int, x: None) -> None:
    ...


def _ratio_to_size(total: int, x: Optional[Union[int, float, Fraction]]) -> Optional[int]:
    return int(x * total) if isinstance(x, (float, Fraction)) else x


@wrap_as_partial_transformer
def downscaler(
    image: _VideoFrameOrFrames, factors: Union[int, tuple[int, int]]
) -> _VideoFrameOrFrames:
    # 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape
    # [height, width, channels].
    if image.ndim == 3:
        height = image.shape[0]
        width = image.shape[1]
    elif image.ndim == 4:
        height = image.shape[1]
        width = image.shape[2]
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    height_factor, width_factor = (factors, factors) if isinstance(factors, int) else factors

    new_height = round(height / height_factor)
    new_width = round(width / width_factor)

    return typing.cast(
        _VideoFrameOrFrames,
        tf.image.resize(image, (new_height, new_width), antialias=True).numpy(),
    )


@wrap_as_partial_transformer
def grayscaler(image: _VideoFrameOrFrames) -> _VideoFrameOrFrames:
    if image.ndim not in {3, 4}:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    if image.shape[-1] != 3:
        raise RuntimeError('expected image to contain 3 color channels')

    return typing.cast(_VideoFrameOrFrames, tf.image.rgb_to_grayscale(image).numpy())


@wrap_as_partial_transformer
def random_cropper(
    image: _VideoFrameOrFrames,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> _VideoFrameOrFrames:
    if image.ndim == 3:
        total_height, total_width, number_of_channels = image.shape

        size = (height or total_height, width or total_width, number_of_channels)
    elif image.ndim == 4:
        batch_size, total_height, total_width, number_of_channels = image.shape

        size = (batch_size, height or total_height, width or total_width, number_of_channels)
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    return typing.cast(_VideoFrameOrFrames, tf.image.random_crop(image, size).numpy())


def normalized_mutual_information(
    image0: VideoFrame, image1: VideoFrame, *, bins: int = 100
) -> float:
    """Compute the normalized mutual information (NMI).

    It ranges from 0 (perfectly uncorrelated image values) to 1 (perfectly correlated image values,
    whether positively or negatively).
    """
    return _normalized_mutual_information(image0, image1, bins=bins) - 1


def structural_similarity_ratio(ref: VideoFrame, image: VideoFrame) -> float:
    return structural_similarity(ref, image) / structural_similarity(ref, ref)


def structural_similarity(ref: VideoFrame, image: VideoFrame) -> float:
    WINDOW_SIZE = 11

    ref, image = _reshape_to_largest(ref, image)

    return _structural_similarity(
        np.squeeze(ref),
        np.squeeze(image),
        win_size=WINDOW_SIZE,
    )


def variance(image: VideoFrame, /) -> float:
    return float(np.var(np.squeeze(image)))


def retained_variance(ref: VideoFrame, image: VideoFrame, /) -> float:
    return variance(image) / variance(ref)


def shannon_cross_entropy(
    ref: VideoFrame,
    image: VideoFrame,
    /,
    *,
    nbins: int = 100,
    epsilon: float = 1e-9,
) -> float:
    ref, image = _reshape_to_largest(ref, image)
    ref = np.squeeze(ref)
    image = np.squeeze(image)

    ref_histogram, _ = np.histogram(ref, bins=nbins, density=True)
    img_histogram, _ = np.histogram(image, bins=nbins, density=True)

    ref_histogram = np.clip(ref_histogram, epsilon, 1 - epsilon)
    img_histogram = np.clip(img_histogram, epsilon, 1 - epsilon)

    return -float(np.sum(ref_histogram * np.log(img_histogram)))


def shannon_cross_entropy_ratio(
    ref: VideoFrame,
    image: VideoFrame,
    /,
    *,
    nbins: int = 100,
    epsilon: float = 1e-9,
) -> float:
    return shannon_cross_entropy(ref, image, nbins=nbins, epsilon=epsilon) / shannon_cross_entropy(
        ref, ref, nbins=nbins, epsilon=epsilon
    )


def shannon_entropy_ratio(ref: VideoFrame, image: VideoFrame, /) -> float:
    return typing.cast(float, shannon_entropy(image) / shannon_entropy(ref))


def nbins_retained_variance(ref: VideoFrame, image: VideoFrame, /) -> float:
    ref_hist = _hist(ref)
    image_hist = _hist(image)

    return float(np.var(image_hist - image_hist.mean()) / np.var(ref_hist - ref_hist.mean()))


def nbins_shannon_entropy_ratio(ref: VideoFrame, image: VideoFrame, /) -> float:
    ref_hist = _hist(ref)
    image_hist = _hist(image)

    return (entropy(ref_hist) + entropy(ref_hist, image_hist)) / (
        entropy(ref_hist) + entropy(ref_hist, ref_hist)
    )


def _hist(image: VideoFrame, /) -> np.ndarray:
    hist, _ = np.histogram(np.squeeze(image).flatten(), bins=100, density=True)
    hist /= hist.sum()
    epsilon = np.finfo(float).eps
    hist = np.clip(hist, epsilon, 1 - epsilon)
    return hist


def _reshape_to_largest(
    image0: VideoFrame,
    image1: VideoFrame,
    /,
) -> tuple[VideoFrame, VideoFrame]:
    if image0.shape != image1.shape:
        max_shape = np.maximum(image0.shape, image1.shape)
        image0 = resize(image0, max_shape)
        image1 = resize(image1, max_shape)
    return image0, image1
