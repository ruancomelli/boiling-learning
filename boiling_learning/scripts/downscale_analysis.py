from functools import partial
from pprint import pprint
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imshow

from boiling_learning.preprocessing.image import crop, downscale


def evaluate_downsampling(
    img: np.ndarray,
    evaluator: Callable[[np.ndarray, np.ndarray], float],
    downsamplers: Iterable[Callable[[np.ndarray], np.ndarray]],
):
    return [evaluator(img, downsampler(img)) for downsampler in downsamplers]


def img_variance(img: np.ndarray) -> float:
    return float(np.var(img, axis=(0, 1)))


def img_retained_variance(ref: np.ndarray, img: np.ndarray) -> float:
    return img_variance(img) / img_variance(ref)


def img_shannon_cross_entropy(ref: np.ndarray, img: np.ndarray) -> float:
    pass


def img_shannon_cross_entropy_ratio(ref: np.ndarray, img: np.ndarray) -> float:
    return float(
        img_shannon_cross_entropy(ref, img)
        / img_shannon_cross_entropy(ref, ref)
    )


def main(
    image: np.ndarray,
    downscale_factors: Iterable[int] = range(1, 11),
    final_downscale_factor: int = 5,
    cropper: Callable[[np.ndarray], np.ndarray] = partial(
        crop, top=600, bottom=230, left=1000, right=1100
    ),
) -> None:
    image = rgb2gray(image)
    downscale_factors = tuple(downscale_factors)

    ev_ds = evaluate_downsampling(
        image,
        img_retained_variance,
        [partial(downscale, shape=ds) for ds in downscale_factors],
    )
    print('Downscale factors scores:')
    pprint(dict(zip(downscale_factors, ev_ds)))

    # ------------------------------------------
    # PART 2
    # ------------------------------------------
    plt.subplot(1, 3, 1)

    imshow(image)

    plt.subplot(1, 3, 2)

    cropped = cropper(image)
    print('Cropped shape:', cropped.shape)
    imshow(cropped)

    plt.subplot(1, 3, 3)

    cropped = img_as_float(cropped)
    downscaled = downscale(cropped, final_downscale_factor)
    print('Downscaled shape:', downscaled.shape)
    imshow(downscaled)
