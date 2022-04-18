from __future__ import annotations

import itertools
import operator
import os
import pprint
import random
import string
from collections import ChainMap
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import funcy
import modin.pandas as pd
from sortedcontainers import SortedSet
from typing_extensions import overload

from boiling_learning.utils.iterutils import flaglast

_TypeT = TypeVar('_TypeT', bound=Type[Any])
_T = TypeVar('_T')
_Key = TypeVar('_Key')
_Value = TypeVar('_Value')


# ---------------------------------- Typing ----------------------------------
# see <https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support>
JSONDataType = Union[None, bool, int, float, str, List['JSONDataType'], Dict[str, 'JSONDataType']]

PathLike = Union[str, os.PathLike]


# ---------------------------------- Utility functions ----------------------------------
@overload
def indexify(arg: Collection) -> range:
    ...


@overload
def indexify(arg: Iterable) -> Iterable[int]:
    ...


def indexify(arg: Union[Iterable, Collection]) -> Iterable[int]:
    try:
        return range(len(arg))
    except TypeError:
        return funcy.walk(0, enumerate(arg))


def reorder(seq: Sequence[_T], indices: Iterable[int]) -> Iterable[_T]:
    return map(seq.__getitem__, indices)


def argmin(iterable: Iterable) -> int:
    return min(enumerate(iterable), key=operator.itemgetter(1))[0]


def argmax(iterable: Iterable) -> int:
    return max(enumerate(iterable), key=operator.itemgetter(1))[0]


def argsorted(iterable: Iterable) -> Iterable[int]:
    return funcy.pluck(0, sorted(enumerate(iterable), key=operator.itemgetter(1)))


def missing_ints(ints: Iterable[int]) -> Iterable[int]:
    # source: adapted from <https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence>
    ints = SortedSet(ints)
    if ints:
        start, end = ints[0], ints[-1]
        full = range(start, end + 1)
        return itertools.filterfalse(ints.__contains__, full)
    else:
        return ()


def merge_dicts(*dict_args: Mapping, latter_precedence: bool = True) -> dict:
    if latter_precedence:
        dict_args = reversed(dict_args)

    return dict(ChainMap(*dict_args))


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


class KeyedDefaultDict(DefaultDict[_Key, _Value]):
    '''

    Source: https://stackoverflow.com/a/2912455/5811400
    '''

    def __missing__(self, key: _Key) -> _Value:
        if self.default_factory is None:
            raise KeyError(key)

        ret = self[key] = self.default_factory(key)
        return ret


def concatenate_dataframes(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate while preserving categorical columns.

    Source: <https://stackoverflow.com/a/57809778/5811400>

    NB: We change the categories in-place for the input dataframes"""

    dfs = tuple(dfs)

    # Iterate on categorical columns common to all dfs
    for col in set().intersection(
        *(set(df.select_dtypes(include='category').columns) for df in dfs)
    ):
        # Generate the union category across dfs for this column
        uc = pd.api.types.union_categoricals(tuple(df[col] for df in dfs))
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    return pd.concat(dfs)


def dataframe_categories_to_int(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    # See <https://www.tensorflow.org/tutorials/load_data/pandas_dataframe> for the reasoning
    # behind this

    if not inplace:
        df = df.copy(deep=True)

    for column in df.select_dtypes(include='category').columns:
        df[column] = pd.Categorical(list(df[column])).codes
    return df


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


# ---------------------------------- Class printing ----------------------------------
def simple_pprinter(names: Optional[Tuple[str, ...]] = None):
    def _simple_pprint(self, obj: Any, stream, indent: int, allowance, context, level):
        """
        Modified from pprint dict https://github.com/python/cpython/blob/3.7/Lib/pprint.py#L194
        """
        # Source: <https://stackoverflow.com/a/52521743/5811400>
        write = stream.write

        class_name = obj.__class__.__name__
        write(f'{class_name}(')

        if names is None:
            obj_items = obj.__dict__.copy().items()
        else:
            if len(names) == 0:
                values = ()
            else:
                values = operator.attrgetter(*names)
                if len(names == 1):
                    values = (values,)
            obj_items = zip(names, values)

        _format_kwarg_dict_items(
            self,
            obj_items,
            stream,
            indent + len(class_name),
            allowance + 1,
            context,
            level,
        )
        write(')')

    return _simple_pprint


def _format_kwarg_dict_items(self, items, stream, indent, allowance, context, level):
    '''
    Modified from pprint dict https://github.com/python/cpython/blob/3.7/Lib/pprint.py#L194
    '''
    write = stream.write

    indent += self._indent_per_level
    delimnl = ',\n' + ' ' * indent
    for last, (key, ent) in flaglast(items):
        write(key)
        write('=')
        self._format(
            ent,
            stream,
            indent + len(key) + 1,
            allowance if last else 1,
            context,
            level,
        )
        if not last:
            write(delimnl)


def simple_pprint_class(cls: _TypeT, *names: str) -> _TypeT:
    pprint.PrettyPrinter._dispatch[cls.__repr__] = simple_pprinter(names)

    # Returning the class allows the use of this function as a decorator
    return cls


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
