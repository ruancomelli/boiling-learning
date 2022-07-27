import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]


def resolve(
    path: PathLike,
    root: Optional[PathLike] = None,
    dir: bool = False,
    parents: bool = False,
) -> Path:
    path = Path(path)

    if root is not None:
        root = resolve(root)
        if not path.is_absolute():
            path = root / path
        elif root not in path.resolve().parents:
            raise ValueError(f'incompatible `root` and `path`: {(root, path)}')

    path = path.resolve()

    if dir:
        path.mkdir(exist_ok=True, parents=True)
    elif parents:
        path.parent.mkdir(exist_ok=True, parents=True)

    return path
