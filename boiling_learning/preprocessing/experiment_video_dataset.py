from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils.collections import KeyedSet


class ExperimentVideoDataset(KeyedSet[str, ExperimentVideo]):
    def __init__(self, experiment_videos: Iterable[ExperimentVideo] = ()) -> None:
        super().__init__(_get_experiment_video_name, experiment_videos)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({sorted(self.keys())})'

    def union(self, *others: Iterable[ExperimentVideo]) -> ExperimentVideoDataset:
        return ExperimentVideoDataset(super().union(*others))


@json.encode.instance(ExperimentVideoDataset)
def _encode_image_dataset(obj: ExperimentVideoDataset) -> list[json.JSONDataType]:
    return json.serialize(sorted(obj, key=_get_experiment_video_name))


@describe.instance(ExperimentVideoDataset)
def _describe_image_dataset(obj: ExperimentVideoDataset) -> list[Path]:
    return describe(sorted(obj, key=_get_experiment_video_name))


def _get_experiment_video_name(experiment_video: ExperimentVideo) -> str:
    return experiment_video.name
