from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import BOILING_CASES

BOILING_DIRECT_DATASETS = tuple(
    get_image_dataset(
        case(),
        transformers=default_boiling_preprocessors(direct_visualization=True),
    )
    for case in BOILING_CASES
)

BOILING_INDIRECT_DATASETS = tuple(
    get_image_dataset(
        case(),
        transformers=default_boiling_preprocessors(direct_visualization=False),
    )
    for case in BOILING_CASES
)
