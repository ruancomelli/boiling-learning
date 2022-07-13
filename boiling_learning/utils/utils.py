from __future__ import annotations

import operator
import os
import random
import string
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import funcy

_T = TypeVar('_T')


# ---------------------------------- Typing ----------------------------------
PathLike = Union[str, os.PathLike]


# ---------------------------------- Utility functions ----------------------------------
def reorder(seq: Sequence[_T], indices: Iterable[int]) -> Iterable[_T]:
    return map(seq.__getitem__, indices)


def argmin(iterable: Iterable[Any]) -> int:
    return min(enumerate(iterable), key=operator.itemgetter(1))[0]


def argmax(iterable: Iterable[Any]) -> int:
    return max(enumerate(iterable), key=operator.itemgetter(1))[0]


def argsorted(iterable: Iterable[Any]) -> Iterable[int]:
    return funcy.pluck(0, sorted(enumerate(iterable), key=operator.itemgetter(1)))


def one_factor_at_a_time(
    iterables: Iterable[Iterable],
    default_indices: Iterable[int] = tuple(),
    skip_repeated: bool = True,
) -> Iterator[tuple]:
    '''
    >>> list(one_factor_at_a_time(
    ...     [
    ...         [1, 2, 3],
    ...         'rohan',
    ...         ['alpha', 'beta']
    ...     ],
    ...     default_indices=[1, 3] # equivalent to default_indices = (1, 3, 0)
    ... ))
    [(2, 'a', 'alpha'), (1, 'a', 'alpha'), (3, 'a', 'alpha'), (2, 'r', 'alpha'), (2, 'o', 'alpha'), (2, 'h', 'alpha'), (2, 'n', 'alpha'), (2, 'a', 'beta')]
    '''
    default_indices = tuple(default_indices)
    default_indices = default_indices + (0,) * (len(iterables) - len(default_indices))

    if skip_repeated:
        yield tuple(iterable[default_indices[idx]] for idx, iterable in enumerate(iterables))

    for iterable_index, iterable in enumerate(iterables):
        head = tuple(
            item[head_idx]
            for item, head_idx in zip(iterables[:iterable_index], default_indices[:iterable_index])
        )
        tail = tuple(
            tail[tail_idx]
            for tail, tail_idx in zip(
                iterables[iterable_index + 1 :],
                default_indices[iterable_index + 1 :],
            )
        )
        if skip_repeated:
            for idx, item in enumerate(iterable):
                if idx != default_indices[iterable_index]:
                    yield head + (item,) + tail
        else:
            for item in iterable:
                yield head + (item,) + tail


# ---------------------------------- Path ----------------------------------
def resolve(
    path: PathLike, root: Optional[PathLike] = None, dir: bool = False, parents: bool = False
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


def generate_string(length: int = 6, chars: Sequence[str] = string.ascii_lowercase) -> str:
    '''source: <https://stackoverflow.com/a/2257449/5811400>'''
    # TODO: maybe replace with uuid.uuid4?
    return ''.join(random.choices(chars, k=length))


@contextmanager
def tempdir(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[PathLike] = None,
) -> Iterator[Path]:
    if dir is not None:
        dir = resolve(dir)

    with TemporaryDirectory(suffix=suffix, prefix=prefix, dir=dir) as dirpath:
        yield resolve(dirpath)


@contextmanager
def tempfilepath(suffix: Optional[str] = None) -> Iterator[Path]:
    with tempdir() as dirpath:
        filepath: Path = dirpath / generate_string()

        if suffix is not None:
            filepath = filepath.with_suffix(suffix)

        yield filepath


# ---------------------------------- Mixins ----------------------------------
class SimpleRepr:
    """A mixin implementing a simple __repr__."""

    # Source: <https://stackoverflow.com/a/44595303/5811400>

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        address = id(self) & 0xFFFFFF
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())

        return f'<{class_name} @{address:x} {attrs}>'


class SimpleStr:
    def __str__(self) -> str:
        class_name = self.__class__.__name__
        attrs = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())

        return f'{class_name}({attrs})'
