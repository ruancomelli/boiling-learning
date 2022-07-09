import typing
from fractions import Fraction
from typing import Optional, Tuple, TypeVar, Union, overload

import numpy as np
import tensorflow as tf
from skimage.exposure import histogram
from skimage.measure import shannon_entropy
from skimage.metrics import normalized_mutual_information as _normalized_mutual_information
from skimage.metrics import structural_similarity as ssim
from skimage.transform import downscale_local_mean as _downscale
from skimage.transform import resize

from boiling_learning.preprocessing.transformers import Operator
from boiling_learning.preprocessing.video import VideoFrame, VideoFrames
from boiling_learning.utils.functional import P

VideoFrameOrFrames = Union[VideoFrame, VideoFrames]
_VideoFrameOrFrames = TypeVar('_VideoFrameOrFrames', bound=VideoFrameOrFrames)


class Grayscaler(Operator[VideoFrameOrFrames]):
    def __init__(self) -> None:
        super().__init__(grayscale, P())


class ImageNormalizer(Operator[VideoFrameOrFrames]):
    def __init__(self) -> None:
        super().__init__(normalize_image, P())


class Downscaler(Operator[VideoFrameOrFrames]):
    def __init__(self, factors: Union[int, Tuple[int, int]]) -> None:
        super().__init__(downscale, pack=P(factors=factors))


class Cropper(Operator[VideoFrameOrFrames]):
    def __init__(
        self,
        left: Optional[Union[int, float, Fraction]] = None,
        right: Optional[Union[int, float, Fraction]] = None,
        right_border: Optional[Union[int, float, Fraction]] = None,
        width: Optional[Union[int, float, Fraction]] = None,
        top: Optional[Union[int, float, Fraction]] = None,
        bottom: Optional[Union[int, float, Fraction]] = None,
        bottom_border: Optional[Union[int, float, Fraction]] = None,
        height: Optional[Union[int, float, Fraction]] = None,
    ) -> None:
        pack = P(
            **{
                key: value
                for key, value in {
                    'left': left,
                    'right': right,
                    'right_border': right_border,
                    'width': width,
                    'top': top,
                    'bottom': bottom,
                    'bottom_border': bottom_border,
                    'height': height,
                }.items()
                # remove `None`s to avoid polluting the `pack` argument
                if value is not None
            }
        )

        super().__init__(crop, pack=pack)


class ConvertImageDType(Operator[VideoFrameOrFrames]):
    def __init__(self, dtype: str) -> None:
        super().__init__(convert_image_dtype, P(dtype=dtype))


def convert_image_dtype(image: _VideoFrameOrFrames, *, dtype: str) -> _VideoFrameOrFrames:
    return tf.image.convert_image_dtype(image, tf.dtypes.as_dtype(dtype)).numpy()


class RandomCropper(Operator[VideoFrameOrFrames]):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        pack = P(
            **{
                key: value
                for key, value in {'width': width, 'height': height}.items()
                # remove `None`s to avoid polluting the `pack` argument
                if value is not None
            }
        )

        super().__init__(random_crop, pack=pack)


def crop(
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
        total_height = image.shape[0]
        total_width = image.shape[1]
    elif image.ndim == 4:
        total_height = image.shape[1]
        total_width = image.shape[2]
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    # axis=-1 is the color channel
    # axis=-2 is the horizontal axis
    # axis=-3 is the vertical axis
    left = _ratio_to_size(image.shape[-2], left)
    right = _ratio_to_size(image.shape[-2], right)
    right_border = _ratio_to_size(image.shape[-2], right_border)
    width = _ratio_to_size(image.shape[-2], width)
    top = _ratio_to_size(image.shape[-3], top)
    bottom = _ratio_to_size(image.shape[-3], bottom)
    bottom_border = _ratio_to_size(image.shape[-3], bottom_border)
    height = _ratio_to_size(image.shape[-3], height)

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


@overload
def _ratio_to_size(total: int, x: Union[int, float, Fraction]) -> int:
    ...


@overload
def _ratio_to_size(total: int, x: None) -> None:
    ...


def _ratio_to_size(total: int, x: Optional[Union[int, float, Fraction]]) -> Optional[int]:
    return int(x * total) if isinstance(x, (float, Fraction)) else x


def downscale(
    image: _VideoFrameOrFrames, factors: Union[int, Tuple[int, int]]
) -> _VideoFrameOrFrames:
    # 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape
    # [height, width, channels].

    if isinstance(factors, int):
        height_factor, width_factor = factors, factors
    else:
        height_factor, width_factor = factors

    CHANNEL_FACTOR = 1
    if image.ndim == 3:
        downscale_factors = (height_factor, width_factor, CHANNEL_FACTOR)
    elif image.ndim == 4:
        BATCH_FACTOR = 1
        downscale_factors = (BATCH_FACTOR, height_factor, width_factor, CHANNEL_FACTOR)
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    return typing.cast(_VideoFrameOrFrames, _downscale(image, downscale_factors))


def _downscale_tf(
    image: _VideoFrameOrFrames, factors: Union[int, Tuple[int, int]]
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

    if isinstance(factors, int):
        height_factor, width_factor = factors, factors
    else:
        height_factor, width_factor = factors

    return typing.cast(
        _VideoFrameOrFrames,
        tf.image.resize(
            image,
            (height // height_factor, width // width_factor),
        ).numpy(),
    )


def grayscale(image: _VideoFrameOrFrames) -> _VideoFrameOrFrames:
    if image.ndim not in {3, 4}:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    if image.shape[-1] != 3:
        raise RuntimeError('expected image to contain 3 color channels')

    return typing.cast(_VideoFrameOrFrames, tf.image.rgb_to_grayscale(image).numpy())


def normalize_image(image: _VideoFrameOrFrames) -> _VideoFrameOrFrames:
    return typing.cast(_VideoFrameOrFrames, tf.image.per_image_standardization(image).numpy())


def random_crop(
    image: _VideoFrameOrFrames,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> _VideoFrameOrFrames:
    if image.ndim == 3:
        total_height = image.shape[0]
        total_width = image.shape[1]
        number_of_channels = image.shape[2]

        size = (height or total_height, width or total_width, number_of_channels)
    elif image.ndim == 4:
        batch_size = image.shape[0]
        total_height = image.shape[1]
        total_width = image.shape[2]
        number_of_channels = image.shape[3]

        size = (batch_size, height or total_height, width or total_width, number_of_channels)
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    return typing.cast(_VideoFrameOrFrames, tf.image.random_crop(image, size).numpy())


def normalized_mutual_information(
    image0: VideoFrame, image1: VideoFrame, *, bins: int = 100
) -> float:
    """Compute the normalized mutual information (NMI).

    The normalized mutual information of :math:`A` and :math:`B` is given by::
    ..math::
        Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}
    where :math:`H(X) := - \sum_{x \in X}{x \log x}` is the entropy.
    It was proposed to be useful in registering images by Colin Studholme and
    colleagues [1]_. It ranges from 0 (perfectly uncorrelated image values)
    to 1 (perfectly correlated image values, whether positively or negatively).
    Parameters
    ----------
    image0, image1 : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.
    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.
    Raises
    ------
    ValueError
        If the images don't have the same number of dimensions.
    Notes
    -----
    If the two input images are not the same shape, the smaller image is
    resized to match the larger one.
    References
    ----------
    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap
           invariant entropy measure of 3D medical image alignment.
           Pattern Recognition 32(1):71-86
           :DOI:`10.1016/S0031-3203(98)00091-0`
    """
    return _normalized_mutual_information(image0, image1, bins=bins) - 1


def structural_similarity_ratio(ref: VideoFrame, image: VideoFrame) -> float:
    # TODO: use https://www.tensorflow.org/api_docs/python/tf/image/ssim?
    # see
    # <https://www.wikiwand.com/en/Structural_similarity#/Application_of_the_formula>
    WINDOW_SIZE = 11

    ref, image = _reshape_to_largest(ref, image)
    ref = np.squeeze(ref)
    image = np.squeeze(image)

    return typing.cast(
        float, ssim(ref, image, win_size=WINDOW_SIZE) / ssim(ref, ref, win_size=WINDOW_SIZE)
    )


def variance(image: VideoFrame) -> float:
    return float(np.var(image, axis=(0, 1)))


def retained_variance(ref: VideoFrame, image: VideoFrame) -> float:
    return variance(image) / variance(ref)


def shannon_cross_entropy(
    ref: VideoFrame, image: VideoFrame, nbins: int = 100, epsilon: float = 1e-9
) -> float:
    ref, image = _reshape_to_largest(ref, image)
    ref = np.squeeze(ref)
    image = np.squeeze(image)

    ref_histogram, _ = histogram(ref, nbins=nbins, normalize=True)
    img_histogram, _ = histogram(image, nbins=nbins, normalize=True)

    ref_histogram = np.clip(ref_histogram, epsilon, 1 - epsilon)
    img_histogram = np.clip(img_histogram, epsilon, 1 - epsilon)

    return -float(np.sum(ref_histogram * np.log(img_histogram)))


def shannon_cross_entropy_ratio(
    ref: VideoFrame, image: VideoFrame, nbins: int = 100, epsilon: float = 1e-9
) -> float:
    return shannon_cross_entropy(ref, image, nbins=nbins, epsilon=epsilon) / shannon_cross_entropy(
        ref, ref, nbins=nbins, epsilon=epsilon
    )


def shannon_entropy_ratio(ref: VideoFrame, image: VideoFrame) -> float:
    return typing.cast(float, shannon_entropy(image) / shannon_entropy(ref))


def _reshape_to_largest(image0: VideoFrame, image1: VideoFrame) -> Tuple[VideoFrame, VideoFrame]:
    if image0.shape != image1.shape:
        max_shape = np.maximum(image0.shape, image1.shape)
        image0 = resize(image0, max_shape)
        image1 = resize(image1, max_shape)
    return image0, image1
