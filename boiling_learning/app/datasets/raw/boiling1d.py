from functools import cache
from pathlib import Path

from loguru import logger

from boiling_learning.app.paths import data_path
from boiling_learning.lazy import LazyCallable, LazyDescribed
from boiling_learning.preprocessing.cases import Case
from boiling_learning.utils.pathutils import PathLike


def boiling_data_path() -> Path:
    return data_path() / 'boiling1d'


def _case_from_path(path: PathLike, /) -> LazyDescribed[Case]:
    return LazyDescribed(
        _load_case_from_path(path),
        # type-ignore is necessary until the classes plugin works again
        path,  # type: ignore[arg-type]
    )


@LazyCallable
def _load_case_from_path(path: PathLike, /) -> Case:
    logger.info(f'Loading boiling case from {path}')
    return Case(path, video_suffix='.MP4').convert_videos(
        '.mp4',
        'converted',
        overwrite=False,
    )


@cache
def boiling_cases() -> tuple[LazyDescribed[Case], ...]:
    return tuple(
        _case_from_path(boiling_data_path() / case_name)
        for case_name in ('case 1', 'case 2', 'case 3', 'case 4')
    )
