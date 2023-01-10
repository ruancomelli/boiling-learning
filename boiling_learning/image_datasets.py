from typing import TypeAlias

from boiling_learning.datasets.sliceable import SupervisedSliceableDataset
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.preprocessing.video import VideoFrame, VideoFrames

Image: TypeAlias = VideoFrame
Images: TypeAlias = VideoFrames
Targets: TypeAlias = dict[str, float]
ImageDataset: TypeAlias = SupervisedSliceableDataset[Image, Targets]
ImageDatasetTriplet: TypeAlias = DatasetTriplet[ImageDataset]
