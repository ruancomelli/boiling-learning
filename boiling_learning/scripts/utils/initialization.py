from pathlib import Path
from typing import Iterable

from loguru import logger


def check_all_paths_exist(named_paths: Iterable[tuple[str, Path]]) -> None:
    for name, path in named_paths:
        if not path.exists():
            raise RuntimeError(f'path to "{name}" does not exist: {path}')

        logger.info(f'Path for "{name}": {path}')
