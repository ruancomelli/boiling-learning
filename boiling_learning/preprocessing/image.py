import math
from typing import Any, Callable, Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf
from decorator import decorator
from scipy.stats import entropy
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

_T = TypeVar('_T')
# something convertible to tf.Tensor
ImageType = tf.types.experimental.TensorLike
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


def reshape_to_largest(image0: np.ndarray, image1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image0.shape != image1.shape:
        max_shape = np.maximum(image0.shape, image1.shape)
        image0 = resize(image0, max_shape)
        image1 = resize(image1, max_shape)
    return image0, image1


def ensure_tensor(image: ImageType, *, dtype: Optional[tf.DType] = None) -> tf.Tensor:
    image = tf.convert_to_tensor(image)
    if dtype is not None:
        image = tf.cast(image, dtype)
    return image


def autocast(dtype: tf.DType) -> Callable[[CallableT], CallableT]:
    @decorator
    def _autocast(
        f: Callable[..., _T],
        image: ImageType,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        return f(ensure_tensor(image, dtype=dtype), *args, **kwargs)

    return _autocast


def ensure_grayscale(image: ImageType) -> np.ndarray:
    return rgb2gray(image) if image.shape[2] != 1 else image


@autocast(tf.float64)
def downscale(image: ImageType, factors: Tuple[int, int], antialias: bool = False) -> tf.Tensor:
    sizes = (
        math.ceil(image.shape[0] / factors[0]),
        math.ceil(image.shape[1] / factors[1]),
    )
    return tf.image.resize(image, sizes, method='bilinear', antialias=antialias)


def normalized_mutual_information(
    image0: np.ndarray, image1: np.ndarray, bins: int = 100
) -> float:
    r"""Compute the normalized mutual information (NMI).

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

    image0, image1 = reshape_to_largest(image0, image1)

    hist, _ = np.histogramdd(
        [np.reshape(image0, -1), np.reshape(image1, -1)],
        bins=bins,
        density=True,
    )

    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))

    return (H0 + H1) / H01 - 1


def structural_similarity_ratio(ref: np.ndarray, image: np.ndarray) -> float:
    # see
    # <https://www.wikiwand.com/en/Structural_similarity#/Application_of_the_formula>
    WINDOW_SIZE: int = 11

    ref, image = reshape_to_largest(ref, image)
    ref = np.squeeze(ref)
    image = np.squeeze(image)

    return ssim(ref, image, win_size=WINDOW_SIZE) / ssim(ref, ref, win_size=WINDOW_SIZE)


def variance(image: np.ndarray) -> float:
    return float(np.var(image, axis=(0, 1)))


def retained_variance(ref: np.ndarray, image: np.ndarray) -> float:
    return variance(image) / variance(ref)


def shannon_cross_entropy(
    ref: np.ndarray, image: np.ndarray, nbins: int = 100, epsilon: float = 1e-9
) -> float:
    ref, image = reshape_to_largest(ref, image)
    ref = np.squeeze(ref)
    image = np.squeeze(image)

    ref_histogram, _ = histogram(ref, nbins=nbins, normalize=True)
    img_histogram, _ = histogram(image, nbins=nbins, normalize=True)

    ref_histogram = np.clip(ref_histogram, epsilon, 1 - epsilon)
    img_histogram = np.clip(img_histogram, epsilon, 1 - epsilon)

    return -float(np.sum(ref_histogram * np.log(img_histogram)))


def shannon_cross_entropy_ratio(
    ref: np.ndarray, image: np.ndarray, nbins: int = 100, epsilon: float = 1e-9
) -> float:
    return shannon_cross_entropy(ref, image, nbins=nbins, epsilon=epsilon) / shannon_cross_entropy(
        ref, ref, nbins=nbins, epsilon=epsilon
    )


def shannon_entropy_ratio(ref: np.ndarray, image: np.ndarray) -> float:
    return shannon_entropy(image) / shannon_entropy(ref)
