from pathlib import Path
from typing import Iterable

from boiling_learning.utils.utils import PathLike, resolve


def itertree(path: PathLike) -> Iterable[Path]:
    path = resolve(path)

    return filter(Path.is_file, path.rglob('*'))
