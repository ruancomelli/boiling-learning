from fractions import Fraction
from typing import Optional, Tuple, Union, overload

import numpy as np
from skimage.color import rgb2gray as _grayscale
from skimage.transform import downscale_local_mean as _downscale


@overload
def _ratio_to_size(
    image: np.ndarray, x: Union[int, float, Fraction], *, axis: int
) -> int:
    ...


@overload
def _ratio_to_size(image: np.ndarray, x: None, *, axis: int) -> None:
    ...


def _ratio_to_size(
    image: np.ndarray, x: Optional[Union[int, float, Fraction]], *, axis: int
) -> Optional[int]:
    if isinstance(x, (float, Fraction)):
        return int(x * image.shape[axis])
    else:
        return x


def _crop(
    image: np.ndarray,
    *,
    left: Optional[int],
    right: Optional[int],
    top: Optional[int],
    bottom: Optional[int],
) -> np.ndarray:
    return image[top:bottom, left:right, ...]


def crop(
    image: np.ndarray,
    *,
    left: Optional[Union[int, float, Fraction]] = None,
    right: Optional[Union[int, float, Fraction]] = None,
    right_border: Optional[Union[int, float, Fraction]] = None,
    width: Optional[Union[int, float, Fraction]] = None,
    top: Optional[Union[int, float, Fraction]] = None,
    bottom: Optional[Union[int, float, Fraction]] = None,
    bottom_border: Optional[Union[int, float, Fraction]] = None,
    height: Optional[Union[int, float, Fraction]] = None,
) -> np.ndarray:
    total_height: int = image.shape[0]
    total_width: int = image.shape[1]

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
    incompatible_pair_error_message = (
        'at least one of `{}` and `{}` must be `None` or omitted.'
    )

    if (left, right, right_border, width).count(None) < 2:
        raise TypeError(
            incompatible_arguments_error_message.format(
                'left', 'right', 'right_border', 'width'
            )
        )

    if None not in {right, right_border}:
        raise TypeError(
            incompatible_pair_error_message.format('right', 'right_border')
        )

    if (top, bottom, bottom_border, height).count(None) < 2:
        raise TypeError(
            incompatible_arguments_error_message.format(
                'top', 'bottom', 'bottom_border', 'height'
            )
        )

    if None not in {bottom, bottom_border}:
        raise TypeError(
            incompatible_pair_error_message.format('bottom', 'bottom_border')
        )

    if right_border is not None:
        right = total_width - right_border
    if bottom_border is not None:
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

    return _crop(image, left=left, right=right, top=top, bottom=bottom)


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim > 2 and image.shape[2] != 1:
        return _grayscale(image)
    return image


def downscale(
    image: np.ndarray, factors: Union[int, Tuple[int, int]]
) -> np.ndarray:
    if isinstance(factors, int):
        factors = (factors, factors)

    return _downscale(image, factors)
