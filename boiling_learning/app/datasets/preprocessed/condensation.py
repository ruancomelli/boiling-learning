from functools import cache

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_condensation_preprocessors
from boiling_learning.app.datasets.raw.condensation import condensation_datasets
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.transforms import datasets_concatenater

# TODO: do I need to subsample this??


@cache
def condensation_dataset() -> LazyDescribed[ImageDatasetTriplet]:
    preprocessors = default_condensation_preprocessors(
        downscale_factor=5,
        height=8 * 12,
        width=8 * 12,
    )
    return (
        LazyDescribed.from_describable(
            tuple(
                get_image_dataset(
                    ds(),
                    preprocessors,
                    experiment='condensation',
                )
                for ds in condensation_datasets()
            )
        )
        | datasets_concatenater()
    )
