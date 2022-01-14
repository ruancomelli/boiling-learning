from __future__ import annotations

import enum
import itertools
import json
import operator
import os
import pprint
import random
import string
from collections import ChainMap
from contextlib import contextmanager
from functools import partial
from os.path import relpath as _relative_path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    Callable,
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
import matplotlib.pyplot as plt
import modin.pandas as pd
import more_itertools as mit
from sortedcontainers import SortedSet
from typing_extensions import overload

from boiling_learning.utils.functional import Kwargs
from boiling_learning.utils.iterutils import flaglast

# ---------------------------------- Typing ----------------------------------
_EnumType = TypeVar('_EnumType', bound=enum.Enum)
_TypeT = TypeVar('_TypeT', bound=Type)
_T = TypeVar('_T')
_Key = TypeVar('_Key')
_Value = TypeVar('_Value')
S = TypeVar('S')


# see <https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support>
VerboseType = Union[bool, int]
JSONDataType = Union[
    None,
    bool,
    int,
    float,
    str,
    List['JSONDataType'],
    Dict[str, 'JSONDataType'],
]


PathLike = Union[str, os.PathLike]


# ---------------------------------- Utility functions ----------------------------------
@overload
def indexify(arg: Iterable) -> Iterable[int]:
    ...


@overload
def indexify(arg: Collection) -> range:
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


def invert_dict(d: Mapping) -> dict:
    return funcy.walk(reversed, dict(d))


def extract_keys(
    d: Mapping[_Key, _Value],
    value: _T,
    cmp: Callable[[_T, _Value], bool] = operator.eq,
) -> Iterator[_Key]:
    comparer: Callable[[_Value], bool] = partial(cmp, value)

    for k, v in d.items():
        if comparer(v):
            yield k


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
    for col in set.intersection(
        *(set(df.select_dtypes(include='category').columns) for df in dfs)
    ):
        # Generate the union category across dfs for this column
        uc = pd.api.types.union_categoricals(tuple(df[col] for df in dfs))
        # Change to union category for all dataframes
        for df in dfs:
            df[col] = pd.Categorical(df[col].values, categories=uc.categories)
    return pd.concat(dfs)


def dataframe_categories_to_int(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    # See <https://www.tensorflow.org/tutorials/load_data/pandas_dataframe> for the reasoning behind this

    if not inplace:
        df = df.copy(deep=True)

    for column in df.select_dtypes(include='category').columns:
        df[column] = pd.Categorical(list(df[column])).codes
    return df


# ---------------------------------- Printing ----------------------------------
def print_verbose(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs)


def print_header(
    s: str,
    level: int = 0,
    levels: Sequence[str] = ('#', '=', '-', '~', '^', '*', '+'),
    verbose: bool = True,
) -> None:
    """Standardized method for printing a section header.

    Prints the argument s underlined.

    Parameters
    ----------
    s       : string to be printed
    level   : index of level symbol to be used
    levels  : iterable of level symbols to choose from
    verbose : verbose mode
    """
    if verbose:
        s = str(s)
        print()
        print(s)
        print(levels[level] * len(s))


def shorten_path(path, max_parts=None, max_len=None, prefix='...'):
    def _slice_path(p, slc):
        return Path(*Path(p).parts[slc])

    path = Path(path)

    if max_parts is None:
        shortened = path
    else:
        shortened = _slice_path(path, slice(-max_parts, None))

    if max_len is not None:
        sep = os.sep
        prefix = str(prefix) + sep
        prefix_len = len(prefix)

        while len(str(shortened)) + prefix_len > max_len and len(shortened.parts) > 1:
            shortened = _slice_path(shortened, slice(1, None))

    if shortened == path:
        return str(shortened)
    else:
        return prefix + str(shortened)


# ---------------------------------- Plotting functions ----------------------------------
def prepare_fig(
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    n_elems: Optional[int] = None,
    fig_size: Optional[Union[str, Tuple[int, int]]] = None,
    subfig_size: Optional[Union[str, Tuple[int, int]]] = None,
    tight_layout: bool = True,
) -> dict:
    """Resize figure and calculate the number of rows and columns in the subplot grid.

    Parameters
    ----------
    n_cols       : number of columns in the subplot grid
    n_rows       : number of rows in the subplot grid
    n_elems      : number of elements in the subplot
    fig_size     : total size of the figure
    subfig_size  : total size of each subfigure
    tight_layout : if True, use tight_layout

    Notes
    -----
    * only two of the three arguments n_cols, n_rows and n_elems must be given. The other one is calculated.
    * only two of the two arguments fig_size and subfig_size must be computed. The other one is calculated.
    * fig_size and subfig_size can be a pair (width, height) or a string in ['tiny', 'small', 'normal', 'intermediate', 'large', 'big']
    """
    if (fig_size, subfig_size).count(None) != 1:
        raise ValueError('exactly one of *figsize* and *subfig_size* must be *None*')
    if (n_cols, n_rows, n_elems).count(None) != 1:
        raise ValueError('exactly one of *n_cols*, *n_rows* and *n_elems* must be *None*')

    if n_rows is None:
        n_rows = (n_elems - 1) // n_cols + 1
    elif n_cols is None:
        n_cols = (n_elems - 1) // n_rows + 1
    grid_size = (n_rows, n_cols)

    def validate(size: _T) -> Union[_T, Tuple[int, int]]:
        if size in {'micro'}:
            return (2, 1.5)
        if size in {'tiny'}:
            return (4, 3)
        if size in {'small'}:
            return (7, 5)
        elif size in {'normal', 'intermediate'}:
            return (9, 7)
        elif size in {'large', 'big'}:
            return (18, 15)
        else:
            return size

    if subfig_size is None:
        fig_size = validate(fig_size)
    else:
        subfig_size = validate(subfig_size)
        fig_size = (
            grid_size[1] * subfig_size[0],
            grid_size[0] * subfig_size[1],
        )

    plt.rcParams['figure.figsize'] = fig_size
    if tight_layout:
        plt.tight_layout()

    return {
        'fig_size': fig_size,
        'subfig_size': subfig_size,
        'grid_size': grid_size,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'n_elems': n_elems,
    }


def json_equivalent(
    lhs,
    rhs,
    encoder: Optional[Type] = None,
    decoder: Optional[Type] = None,
    dumps: Callable[[_T], str] = json.dumps,
    loads: Callable[[str], Any] = json.loads,
) -> bool:
    # ignore parameter *cls* when it is *None*
    dumps = Kwargs({'cls': encoder}).partial(dumps)
    loads = Kwargs({'cls': decoder}).partial(loads)

    lhs_str = dumps(lhs)
    rhs_str = dumps(rhs)

    return (lhs_str == rhs_str) or (loads(lhs_str) == loads(rhs_str))


# ---------------------------------- Iteration ----------------------------------
def append(iterable: Iterable[_T], value: S) -> Iterator[Union[_T, S]]:
    yield from iterable
    yield value


def transpose(iterable: _T) -> Iterable[Tuple[_T, ...]]:
    return zip(*iterable)


def replace(iterable, new_iterable):
    # Iterates over iterable and new_iterable, yielding only new_iterable values.
    # This effectively replaces every element in iterable with elements in new_iterable.
    for _, new_value in zip(iterable, new_iterable):
        yield new_value


def drop_last(iterable: Iterable[_T]) -> Iterator[_T]:
    for current_value, _ in mit.pairwise(iterable):
        yield current_value


# ---------------------------------- Path ----------------------------------
def relative_path(origin: PathLike, destination: PathLike) -> Path:
    return _relative_path(resolve(destination), start=resolve(origin))


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


def ensure_dir(path: PathLike, root: Optional[PathLike] = None) -> Path:
    return resolve(path, root=root, dir=True)


def ensure_parent(path: PathLike, root: Optional[PathLike] = None) -> Path:
    return resolve(path, root=root, parents=True)


# Source: https://stackoverflow.com/a/34236245/5811400
def is_parent_dir(parent: PathLike, subdir: PathLike) -> bool:
    parent = resolve(parent)
    subdir = resolve(subdir)
    return parent in subdir.parents


# Source: https://stackoverflow.com/a/57892171/5811400
def rmdir(
    path: PathLike,
    recursive: bool = False,
    missing_ok: bool = False,
) -> None:
    path = resolve(path)

    if not path.is_dir():
        if missing_ok:
            return
        else:
            raise NotADirectoryError(
                f'path is expected to be a directory when missing_ok={missing_ok}. Got {path}'
            )

    if recursive:
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                rmdir(child, recursive=recursive, missing_ok=True)

    path.rmdir()


def dir_as_tree(dir_path, file_pred=None, dir_pred=None):
    ret_list = []
    ret_dict = {}

    for path in dir_path.iterdir():
        if path.is_file() and (file_pred is None or file_pred(path)):
            ret_list.append(path)
        elif path.is_dir() and (dir_pred is None or dir_pred(path)):
            ret_dict[path.name] = dir_as_tree(path, file_pred=file_pred, dir_pred=dir_pred)

    return ret_list, ret_dict


def dir_as_tree_apply(dir_path, fs, dir_pred=None):
    return [f(dir_path) for f in fs], {
        path.name: dir_as_tree_apply(path, fs, dir_pred=dir_pred)
        for path in dir_path.iterdir()
        if path.is_dir() and (dir_pred is None or dir_pred(path))
    }


def generate_string(length: int = 6, chars: Sequence[str] = string.ascii_lowercase) -> str:
    '''source: <https://stackoverflow.com/a/2257449/5811400>'''
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
        write(class_name + '(')

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


class DictEq:
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return vars(self) == vars(other)

    def __ne__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        return not self.__eq__(other)


# ---------------------------------- Enum ----------------------------------
def enum_item(enumeration: Type[_EnumType], item: Union[_EnumType, int, str]) -> _EnumType:
    if isinstance(item, str):
        return enumeration[item]
    elif isinstance(item, int):
        return enumeration(item)
    else:
        return item


# ---------------------------------- Operator ----------------------------------
def contains(elem):
    '''Return a predicated that tests if a container contains elem.'''
    return funcy.rpartial(operator.contains, elem)


def is_(x) -> Callable[[Any], bool]:
    return partial(operator.is_, x)


def is_not(x) -> Callable[[Any], bool]:
    return partial(operator.is_not, x)
