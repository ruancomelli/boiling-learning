from fractions import Fraction
from functools import cache
from typing import Literal

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import (
    DEFAULT_DOWNSCALE_FACTOR,
    DEFAULT_VISUALIZATION_WINDOW_WIDTH,
    DEFAULT_WIDTH,
    RECOMMENDED_DOWNSCALE_FACTOR,
    RECOMMENDED_VISUALIZATION_WINDOW_WIDTH,
    RECOMMENDED_WIDTH,
    default_boiling_preprocessors,
)
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed


@cache
def boiling_datasets(
    *,
    direct_visualization: bool = True,
    downscale_factor: int = DEFAULT_DOWNSCALE_FACTOR,
    height: int | None = None,
    bottom_border: int | None = None,
    width: int = DEFAULT_WIDTH,
    visualization_window_width: Fraction = DEFAULT_VISUALIZATION_WINDOW_WIDTH,
    crop_mode: Literal["center", "random"] = "center",
    shuffle: bool = True,
) -> tuple[LazyDescribed[ImageDatasetTriplet], ...]:
    return tuple(
        get_image_dataset(
            case(),
            transformers=default_boiling_preprocessors(
                direct_visualization=direct_visualization,
                downscale_factor=downscale_factor,
                height=height,
                bottom_border=bottom_border,
                width=width,
                visualization_window_width=visualization_window_width,
                crop_mode=crop_mode,
            ),
            experiment="boiling1d",
            shuffle=shuffle,
        )
        for case in boiling_cases()
    )


def baseline_boiling_dataset(
    *,
    direct_visualization: bool,
) -> LazyDescribed[ImageDatasetTriplet]:
    return boiling_datasets(
        direct_visualization=direct_visualization,
        downscale_factor=RECOMMENDED_DOWNSCALE_FACTOR,
        width=RECOMMENDED_WIDTH,
        visualization_window_width=RECOMMENDED_VISUALIZATION_WINDOW_WIDTH,
        crop_mode="center",
    )[0]
