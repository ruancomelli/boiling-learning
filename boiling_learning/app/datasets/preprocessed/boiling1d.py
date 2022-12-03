from functools import cache

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed


@cache
def boiling_datasets(
    *,
    direct_visualization: bool,
) -> tuple[LazyDescribed[ImageDatasetTriplet], ...]:
    return tuple(
        get_image_dataset(
            case(),
            transformers=default_boiling_preprocessors(direct_visualization=direct_visualization),
            experiment='boiling1d',
        )
        for case in boiling_cases()
    )
