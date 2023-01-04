from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.io.storage import Metadata, deserialize, load, save, serialize
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


@serialize.instance(ExperimentVideoDataset)
def _serialize_experiment_video(instance: ExperimentVideoDataset, path: Path) -> None:
    save(list(instance), path)


@deserialize.dispatch(ExperimentVideoDataset)
def _deserialize_experiment_video(path: Path, _metadata: Metadata) -> ExperimentVideoDataset:
    return ExperimentVideoDataset(load(path))


def _get_experiment_video_name(experiment_video: ExperimentVideo) -> str:
    return experiment_video.name
