from typing import Any, Dict

from typing_extensions import TypeAlias

from boiling_learning.datasets.datasets import DatasetTriplet
from boiling_learning.datasets.sliceable import SupervisedSliceableDataset
from boiling_learning.preprocessing.video import VideoFrame

Image: TypeAlias = VideoFrame
Targets: TypeAlias = Dict[str, Any]
ImageDataset: TypeAlias = SupervisedSliceableDataset[Image, Targets]
ImageDatasetTriplet: TypeAlias = DatasetTriplet[ImageDataset]
