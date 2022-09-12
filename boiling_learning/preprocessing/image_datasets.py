from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Callable, List, Mapping

from boiling_learning.describe.describers import describe
from boiling_learning.io import json
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.dataclasses import dataclass, dataclass_from_mapping
from boiling_learning.utils.pathutils import PathLike, resolve


class ImageDataset(KeyedSet[str, ExperimentVideo]):
    @dataclass(frozen=True)
    class VideoDataKeys(ExperimentVideo.VideoDataKeys):
        name: str = 'name'
        ignore: str = 'ignore'

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
        *others: ImageDataset,
        namer: Callable[[List[str]], str] = '+'.join,
    ) -> ImageDataset:
        name = namer([other.name for other in others])
        image_dataset = ImageDataset(name)
        image_dataset.update(*others)
        return image_dataset

    def set_video_data(
        self,
        video_data: Mapping[str, ExperimentVideo.VideoData],
        *,
        remove_absent: bool = False,
    ) -> None:
        video_data_keys = frozenset(video_data.keys())
        self_keys = frozenset(self.keys())
        for name in self_keys & video_data_keys:
            self[name].data = video_data[name]

        if remove_absent:
            for name in self_keys - video_data_keys:
                del self[name]

    def set_video_data_from_file(
        self,
        data_path: PathLike,
        *,
        remove_absent: bool = False,
        keys: VideoDataKeys = VideoDataKeys(),
    ) -> None:
        data_path = resolve(data_path)
        video_data = _json.loads(data_path.read_text())

        if isinstance(video_data, list):
            video_data = {
                item.pop(keys.name): item
                for item in video_data
                if not item.pop(keys.ignore, False)
            }
        elif isinstance(video_data, dict):
            video_data = {
                key: value
                for key, value in video_data.items()
                if not value.pop(keys.ignore, False)
            }
        else:
            raise RuntimeError(f'could not load video data from {data_path}. Got {video_data!r}.')

        video_data = {
            name: dataclass_from_mapping(data, ExperimentVideo.VideoData, key_map=keys)
            for name, data in video_data.items()
        }

        self.set_video_data(video_data, remove_absent=remove_absent)


@json.encode.instance(ImageDataset)
def _encode_image_dataset(obj: ImageDataset) -> List[json.JSONDataType]:
    return json.serialize(sorted(obj, key=_get_experiment_video_name))


@describe.instance(ImageDataset)
def _describe_image_dataset(obj: ImageDataset) -> List[Path]:
    return describe(sorted(obj, key=_get_experiment_video_name))


def _get_experiment_video_name(experiment_video: ExperimentVideo) -> str:
    return experiment_video.name
