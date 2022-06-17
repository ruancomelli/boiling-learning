import typing
from fractions import Fraction
from typing import Optional, Sequence, Tuple, Union, overload

import albumentations as A
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from skimage.exposure import histogram
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.functional import P

VideoFrames = Sequence[VideoFrame]


class Grayscaler(Transformer[VideoFrame, VideoFrame]):
    def __init__(self) -> None:
        super().__init__('grayscale', grayscale, P())


class ImageNormalizer(Transformer[VideoFrame, VideoFrame]):
    def __init__(self) -> None:
        super().__init__('normalize_image', normalize_image, P())


class Downscaler(Transformer[VideoFrame, VideoFrame]):
    def __init__(self, factors: Union[int, Tuple[int, int]]) -> None:
        super().__init__('downscaler', downscale, pack=P(factors=factors))


class Cropper(Transformer[VideoFrame, VideoFrame]):
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

        super().__init__('cropper', crop, pack=pack)


class RandomCropper(Transformer[VideoFrame, VideoFrame]):
    def __init__(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        pack = P(
            **{
                key: value
                for key, value in {'width': width, 'height': height}.items()
                # remove `None`s to avoid polluting the `pack` argument
                if value is not None
            }
        )

        super().__init__('random_cropper', random_crop, pack=pack)


@dataclass
class Shape:
    height: int
    width: int


def shape(image: VideoFrame) -> Shape:
    return Shape(height=image.shape[0], width=image.shape[1])


@overload
def _ratio_to_size(image: VideoFrame, x: Union[int, float, Fraction], *, axis: int) -> int:
    ...


@overload
def _ratio_to_size(image: VideoFrame, x: None, *, axis: int) -> None:
    ...


def _ratio_to_size(
    image: VideoFrame, x: Optional[Union[int, float, Fraction]], *, axis: int
) -> Optional[int]:
    return int(x * image.shape[axis]) if isinstance(x, (float, Fraction)) else x


def crop(
    image: VideoFrame,
    *,
    left: Optional[Union[int, float, Fraction]] = None,
    right: Optional[Union[int, float, Fraction]] = None,
    right_border: Optional[Union[int, float, Fraction]] = None,
    width: Optional[Union[int, float, Fraction]] = None,
    top: Optional[Union[int, float, Fraction]] = None,
    bottom: Optional[Union[int, float, Fraction]] = None,
    bottom_border: Optional[Union[int, float, Fraction]] = None,
    height: Optional[Union[int, float, Fraction]] = None,
) -> VideoFrame:
    if image.ndim == 3:
        total_height = image.shape[0]
        total_width = image.shape[1]
    elif image.ndim == 4:
        total_height = image.shape[1]
        total_width = image.shape[2]
    else:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    left = _ratio_to_size(image, left, axis=1)
    right = _ratio_to_size(image, right, axis=1)
    right_border = _ratio_to_size(image, right_border, axis=1)
    width = _ratio_to_size(image, width, axis=1)
    top = _ratio_to_size(image, top, axis=0)
    bottom = _ratio_to_size(image, bottom, axis=0)
    bottom_border = _ratio_to_size(image, bottom_border, axis=0)
    height = _ratio_to_size(image, height, axis=0)

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

    return tf.image.crop_to_bounding_box(
        image,
        offset_height=top if top is not None else 0,
        offset_width=left if left is not None else 0,
        target_height=bottom - top,
        target_width=right - left,
    ).numpy()


def downscale(image: VideoFrame, factors: Union[int, Tuple[int, int]]) -> VideoFrame:
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
        VideoFrame,
        tf.image.resize(
            image,
            (height // height_factor, width // width_factor),
        ).numpy(),
    )


def grayscale(image: VideoFrame) -> VideoFrame:
    if image.ndim not in {3, 4}:
        raise RuntimeError(f'image must have either 3 or 4 dimensions, got {image.ndim}')

    if image.shape[-1] != 3:
        raise RuntimeError('expected image to contain 3 color channels')

    return tf.image.rgb_to_grayscale(image).numpy()


def normalize_image(image: VideoFrame) -> VideoFrame:
    return tf.image.per_image_standardization(image).numpy()


def random_crop(
    image: VideoFrame,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> VideoFrame:
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

    return typing.cast(VideoFrame, tf.image.random_crop(image, size).numpy())


def random_flip_left_right(image: VideoFrame) -> VideoFrame:
    return typing.cast(VideoFrame, A.HorizontalFlip(p=0.5).apply(image))


def random_brightness_contrast(
    image: VideoFrame,
    brightness_delta: Union[float, Tuple[float, float]],
    contrast_delta: Union[float, Tuple[float, float]],
) -> VideoFrame:
    return typing.cast(
        VideoFrame,
        A.RandomBrightnessContrast(brightness_delta, contrast_delta, always_apply=True).apply(
            image
        ),
    )


def random_jpeg_quality(image: VideoFrame, min_quality: int, max_quality: int = 100) -> VideoFrame:
    if image.dtype.type is np.float64:
        image = image.astype(np.float32)

    return typing.cast(
        VideoFrame,
        A.ImageCompression(
            min_quality,
            max_quality,
            compression_type=A.ImageCompression.ImageCompressionType.JPEG,
            always_apply=True,
        ).apply(image),
    )


def normalized_mutual_information(
    image0: VideoFrame, image1: VideoFrame, bins: int = 100
) -> float:
    """Compute the normalized mutual information (NMI).

    Source code: https://github.com/scikit-image/scikit-image/blob/8db28f027729a045de1b54b599338d1804a461b3/skimage/metrics/simple_metrics.py#L193-L261

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
    if image0.ndim != image1.ndim:
        raise ValueError(
            'NMI requires images of same number of dimensions. '
            f'Got {image0.ndim}D for `image0` and {image1.ndim}D for `image1`.'
        )

    image0, image1 = _reshape_to_largest(image0, image1)

    hist, _ = np.histogramdd(
        [np.reshape(image0, -1), np.reshape(image1, -1)],
        bins=bins,
        density=True,
    )

    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))

    return typing.cast(float, (H0 + H1) / H01 - 1)


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
