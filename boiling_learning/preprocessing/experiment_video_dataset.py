from __future__ import annotations

from pathlib import Path
from typing import Callable

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils.collections import KeyedSet


class ExperimentVideoDataset(KeyedSet[str, ExperimentVideo]):
    def __init__(self, name: str) -> None:
        super().__init__(_get_experiment_video_name)

        self._name = name

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} name={self.name} experiment_videos={sorted(self.keys())}>'
        )

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def make_union(
        cls,
        *others: ExperimentVideoDataset,
        namer: Callable[[list[str]], str] = '+'.join,
    ) -> ExperimentVideoDataset:
        name = namer([other.name for other in others])
        image_dataset = ExperimentVideoDataset(name)
        image_dataset.update(*others)
        return image_dataset


@json.encode.instance(ExperimentVideoDataset)
def _encode_image_dataset(obj: ExperimentVideoDataset) -> list[json.JSONDataType]:
    return json.serialize(sorted(obj, key=_get_experiment_video_name))


@describe.instance(ExperimentVideoDataset)
def _describe_image_dataset(obj: ExperimentVideoDataset) -> list[Path]:
    return describe(sorted(obj, key=_get_experiment_video_name))


def _get_experiment_video_name(experiment_video: ExperimentVideo) -> str:
    return experiment_video.name
