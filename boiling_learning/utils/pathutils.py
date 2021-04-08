from pathlib import Path
from typing import Iterable

from boiling_learning.utils.utils import PathLike, ensure_resolved


def itertree(path: PathLike) -> Iterable[Path]:
    path = ensure_resolved(path)

    return filter(
        Path.is_file,
        path.rglob('*')
    )
