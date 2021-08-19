from __future__ import annotations

import datetime
import enum
import itertools
import json
import operator
import os
import pprint
import re
import shutil
import tempfile
import zlib
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from functools import partial, wraps
from itertools import product
from pathlib import Path
from timeit import default_timer
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import dataclassy
import funcy
import matplotlib.pyplot as plt
import modin.pandas as pd
import more_itertools as mit
import zict
from dataclassy.dataclass import DataClass
from dataclassy.functions import is_dataclass, is_dataclass_instance
from frozendict import frozendict
from more_itertools import unzip
from plum import dispatch
from sortedcontainers import SortedSet
from typing_extensions import overload

from boiling_learning.utils.functional import P
from boiling_learning.utils.iterutils import flaglast

# ---------------------------------- Typing ----------------------------------
_EnumType = TypeVar('_EnumType', bound=enum.Enum)


class _Sentinel(enum.Enum):
    INSTANCE = enum.auto()

    @classmethod
    def get_instance(cls) -> _Sentinel:
        return cls.INSTANCE


_sentinel = _Sentinel.get_instance()
_T = TypeVar('_T')
S = TypeVar('S')
SentinelOptional = Union[_Sentinel, _T]


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
def empty(*args, **kwargs) -> None:
    pass


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


def constant_factory(value: Callable[[], _T]) -> Callable[..., _T]:
    def _constant(*args, **kwargs):
        return value()

    return _constant


def comment(
    f: Callable, s: str = '', printer: Callable[[str], Any] = print
) -> Callable:
    @wraps(f)
    def wrapped(*args, **kwargs):
        printer(s)
        return f(*args, **kwargs)

    return wrapped


@dispatch
def as_immutable(coll: list):
    return tuple(coll)


@dispatch
def as_immutable(coll: set):
    return frozenset(coll)


@dispatch
def as_immutable(coll: dict):
    return frozendict(coll)


# TODO: in the *_sentinel* case, use *key=as_immutable*. Do it and test!
def remove_duplicates(
    iterable: Iterable[_T],
    key: Union[None, object, Dict, Callable] = _sentinel,
) -> Iterable[_T]:
    # See <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_everseen>

    if key is _sentinel:
        # use default optimization
        return remove_duplicates(
            iterable,
            key={
                list: tuple,
                set: frozenset,
                dict: (lambda elem: frozenset(elem.items())),
            },
        )
    elif isinstance(key, dict):
        return mit.unique_everseen(
            iterable,
            key=lambda elem: key.get(type(elem), funcy.identity)(elem),
        )
    else:
        return mit.unique_everseen(iterable, key=key)


def reorder(seq: Sequence[_T], indices: Iterable[int]) -> Iterable[_T]:
    return map(seq.__getitem__, indices)


def argmin(iterable: Iterable) -> int:
    return min(enumerate(iterable), key=operator.itemgetter(1))[0]


def argmax(iterable: Iterable) -> int:
    return max(enumerate(iterable), key=operator.itemgetter(1))[0]


def argsorted(iterable: Iterable) -> Iterable[int]:
    return funcy.pluck(
        0, sorted(enumerate(iterable), key=operator.itemgetter(1))
    )


def multipop(lst: MutableSequence[_T], indices: Collection[int]) -> List[_T]:
    pop = [lst[i] for i in indices]
    lst[:] = [v for i, v in enumerate(lst) if i not in indices]

    return pop


def has_duplicates(iterable: Iterable) -> bool:
    if isinstance(iterable, Sequence):
        iterable_seq = iterable
    elif isinstance(iterable, set):
        return False
    else:
        iterable_seq = tuple(iterable)

    return len(iterable_seq) != mit.ilen(remove_duplicates(iterable_seq))


def missing_ints(ints: Iterable[int]) -> Iterable[int]:
    # source: adapted from <https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence>
    ints = SortedSet(ints)
    if ints:
        start, end = ints[0], ints[-1]
        full = range(start, end + 1)
        return itertools.filterfalse(ints.__contains__, full)
    else:
        return empty_gen()


def is_consecutive(ints: Iterable[int], ignore_order: bool = False) -> bool:
    ints = tuple(ints)
    if not ints:
        return True

    if ignore_order:
        # Source: https://stackoverflow.com/a/64177833/5811400
        r = range(min(ints), max(ints) + 1)
        return len(ints) == len(r) and funcy.all(ints.__contains__, r)
    else:
        r = range(ints[0], ints[-1] + 1)
        return ints == tuple(r)


def merge_dicts(*dict_args: Mapping, latter_precedence: bool = True) -> dict:
    if latter_precedence:
        dict_args = reversed(dict_args)

    return dict(ChainMap(*dict_args))


def partial_isinstance(type_: Type) -> Callable[[Any], bool]:
    def wrapped(x) -> bool:
        return isinstance(x, type_)

    return wrapped


def one_factor_at_a_time(
    iterables: Iterable[Iterable],
    default_indices: Iterable[int] = tuple(),
    skip_repeated: bool = True,
) -> Iterator[tuple]:
    '''
    >>> one_factor_at_a_time(
        [
            [1, 2, 3],
            'rohan',
            ['alpha', 'beta']
        ],
        default_indices=[1, 3] # equivalent to default_indices = (1, 3, 0)
    )
    [(1, 'a', 'alpha'),
    (2, 'a', 'alpha'),
    (3, 'a', 'alpha'),
    (2, 'r', 'alpha'),
    (2, 'o', 'alpha'),
    (2, 'h', 'alpha'),
    (2, 'a', 'alpha'),
    (2, 'n', 'alpha'),
    (2, 'a', 'alpha'),
    (2, 'a', 'beta')]
    '''
    default_indices = tuple(default_indices)
    default_indices = default_indices + (0,) * (
        len(iterables) - len(default_indices)
    )

    if skip_repeated:
        yield tuple(
            iterable[default_indices[idx]]
            for idx, iterable in enumerate(iterables)
        )

    for iterable_index, iterable in enumerate(iterables):
        head = tuple(
            item[head_idx]
            for item, head_idx in zip(
                iterables[:iterable_index], default_indices[:iterable_index]
            )
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


def combine_dict(
    dct: Mapping[Any, Iterable],
    gen: Optional[Callable[[list], Iterable[Iterable]]] = None,
) -> Iterator[dict]:
    if gen is None:

        def default_gen(x):
            return product(*x)

        gen = default_gen

    keys, iterables = unzip(dct.items())
    keys = list(keys)
    iterables = list(iterables)

    for combination in gen(iterables):
        yield dict(zip(keys, combination))


def dict_product(**kwargs: Mapping) -> Iterator[dict]:
    # source: <https://stackoverflow.com/a/5228294/5811400>
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


def invert_dict(d: Mapping) -> dict:
    return funcy.walk(reversed, dict(d))


class inclusive_bidict(dict):
    '''Inclusive bidirectional dictionary.

    Here is a class for a bidirectional dict.

    Note that:

    1) The inverse directory bd.inverse auto-updates itself when the standard dict bd is modified.
    2) The inverse directory bd.inverse[value] is always a list of key such that bd[key] == value.
    3) Unlike the bidict module from https://pypi.python.org/pypi/bidict, here we can have 2 keys having same value, this is very important.

    Usage:
    >>>> bd = inclusive_bidict({'a': 1, 'b': 2})
    >>>> print(bd)                     # {'a': 1, 'b': 2}
    >>>> print(bd.inverse)             # {1: ['a'], 2: ['b']}
    >>>> bd['c'] = 1                   # Now two keys have the same value (= 1)
    >>>> print(bd)                     # {'a': 1, 'c': 1, 'b': 2}
    >>>> print(bd.inverse)             # {1: ['a', 'c'], 2: ['b']}
    >>>> del bd['c']
    >>>> print(bd)                     # {'a': 1, 'b': 2}
    >>>> print(bd.inverse)             # {1: ['a'], 2: ['b']}
    >>>> del bd['a']
    >>>> print(bd)                     # {'b': 2}
    >>>> print(bd.inverse)             # {2: ['b']}
    >>>> bd['b'] = 3
    >>>> print(bd)                     # {'b': 3}
    >>>> print(bd.inverse)             # {2: [], 3: ['b']}

    Source: <https://stackoverflow.com/a/21894086/5811400>
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super().__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super().__delitem__(key)


def extract_keys(d, value, cmp=operator.eq):
    for k, v in d.items():
        if cmp(value, v):
            yield k


def map_keys(dct, key_map):
    return {v: dct[k] for k, v in key_map.items()}


class KeyedDefaultDict(defaultdict):
    '''

    Source: https://stackoverflow.com/a/2912455/5811400
    '''

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)

        ret = self[key] = self.default_factory(key)
        return ret


def is_dataclass_class(type_) -> bool:
    return is_dataclass(type_) and not is_dataclass_instance(type_)


def dataclass_from_mapping(
    mapping: Mapping[str, Any],
    dataclass_factory: Callable[..., _T],
    key_map: Optional[Union[DataClass, Mapping[str, str]]] = None,
) -> _T:
    if not is_dataclass_class(dataclass_factory):
        raise ValueError('*dataclass_factory* must be a dataclass.')

    dataclass_field_names = frozenset(dataclassy.fields(dataclass_factory))

    if key_map is None:
        return dataclass_factory(
            **funcy.select_keys(dataclass_field_names, mapping)
        )

    if is_dataclass_instance(key_map):
        key_map = dataclassy.as_dict(key_map)

    key_map = funcy.select_keys(dataclass_field_names, key_map)
    translator = invert_dict(key_map).get
    mapping = {translator(key, key): value for key, value in mapping.items()}
    return dataclass_from_mapping(mapping, dataclass_factory)


def to_parent_dataclass(
    obj: DataClass, parent: Callable[..., DataClass]
) -> DataClass:
    if not is_dataclass_class(parent):
        raise ValueError('*parent* must be a dataclass.')

    if not is_dataclass_instance(obj):
        raise ValueError('*obj* must be a dataclass instance.')

    if not isinstance(obj, parent):
        raise ValueError('*obj* must be an instance of *parent*.')

    return dataclass_from_mapping(dataclassy.as_dict(obj), parent)


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


def dataframe_categories_to_int(
    df: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame:
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

        while (
            len(str(shortened)) + prefix_len > max_len
            and len(shortened.parts) > 1
        ):
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
        raise ValueError(
            'exactly one of *figsize* and *subfig_size* must be *None*'
        )
    if (n_cols, n_rows, n_elems).count(None) != 1:
        raise ValueError(
            'exactly one of *n_cols*, *n_rows* and *n_elems* must be *None*'
        )

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
    dumps = P(cls=encoder).partial(dumps)
    loads = P(cls=decoder).partial(loads)

    lhs_str = dumps(lhs)
    rhs_str = dumps(rhs)

    return (lhs_str == rhs_str) or (loads(lhs_str) == loads(rhs_str))


# ---------------------------------- Iteration ----------------------------------
def empty_gen() -> Iterator[None]:
    # Source: <https://stackoverflow.com/a/13243870/5811400>
    return
    yield


def append(iterable: Iterable[_T], value: S) -> Iterator[Union[_T, S]]:
    yield from iterable
    yield value


def transpose(iterable: _T) -> Iterable[Tuple[_T, ...]]:
    return zip(*iterable)


def projection(*indices):
    def wrapped(*args):
        return tuple(args[i] for i in indices)

    return wrapped


def replace(iterable, new_iterable):
    # Iterates over iterable and new_iterable, yielding only new_iterable values.
    # This effectively replaces every element in iterable with elements in new_iterable.
    for _, new_value in zip(iterable, new_iterable):
        yield new_value


def drop_last(iterable):
    for current_value, _ in mit.pairwise(iterable):
        yield current_value


def replace_last(iterable, last_value):
    return append(drop_last(iterable), last_value)


# ---------------------------------- Path ----------------------------------
def relative_path(origin, destination):
    from os.path import relpath

    return relpath(destination, start=origin)


def ensure_resolved(path: PathLike, root: Optional[PathLike] = None) -> Path:
    path = Path(path)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def ensure_dir(path: PathLike, root: Optional[PathLike] = None) -> Path:
    path = ensure_resolved(path, root=root)
    path.mkdir(exist_ok=True, parents=True)
    return path


def ensure_parent(path: PathLike, root: Optional[PathLike] = None) -> Path:
    path = ensure_resolved(path, root=root)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def remove_copy(directory, pattern):
    def remove_copy_idx(path):
        pattern = re.compile(r'(.*) \([0-9]+\)\.png')
        matches = pattern.match(path.name)
        if matches:
            name = matches[1]
            return (True, path.with_name(name).with_suffix(path.suffix))
        else:
            return (False, None)

    for f in Path(directory).glob(pattern):
        success, original_f = remove_copy_idx(f)
        if success and original_f.is_file():
            f.unlink()


# Source: https://stackoverflow.com/a/34236245/5811400
def is_parent_dir(parent: PathLike, subdir: PathLike) -> bool:
    parent = ensure_resolved(parent)
    subdir = ensure_resolved(subdir)
    return parent in subdir.parents


# Source: https://stackoverflow.com/a/57892171/5811400
def rmdir(
    path: PathLike,
    recursive: bool = False,
    keep: bool = False,
    missing_ok: bool = False,
) -> None:
    path = ensure_resolved(path)

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
                rmdir(child, recursive=recursive, keep=False, missing_ok=True)
        if not keep:
            path.rmdir()
    elif keep:
        ValueError('cannot keep dir when not in recursive mode.')


def group_files(path, keyfunc=operator.attrgetter('suffix')):
    d = {}
    for p in filter(operator.methodcaller('is_file'), path.iterdir()):
        d.setdefault(keyfunc(p), []).append(p)
    return d


def dir_as_tree(dir_path, file_pred=None, dir_pred=None):
    ret_list = []
    ret_dict = {}

    for path in dir_path.iterdir():
        if path.is_file() and (file_pred is None or file_pred(path)):
            ret_list.append(path)
        elif path.is_dir() and (dir_pred is None or dir_pred(path)):
            ret_dict[path.name] = dir_as_tree(
                path, file_pred=file_pred, dir_pred=dir_pred
            )

    return ret_list, ret_dict


def dir_as_tree_apply(dir_path, fs, dir_pred=None):
    return [f(dir_path) for f in fs], {
        path.name: dir_as_tree_apply(path, fs, dir_pred=dir_pred)
        for path in dir_path.iterdir()
        if path.is_dir() and (dir_pred is None or dir_pred(path))
    }


def count_file_lines(path):
    with open(path) as f:
        return mit.ilen(f)


@contextmanager
def tempdir(suffix=None, prefix=None, dir=None):
    dirpath = (
        Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir))
        .resolve()
        .absolute()
    )

    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


def JSONDict(
    path: PathLike, dumps: Callable[[_T], str], loads: Callable[[str], _T]
) -> zict.Func:
    path = ensure_dir(path)
    file = zict.File(path, mode='a')
    compress = zict.Func(zlib.compress, zlib.decompress, file)

    return zict.Func(
        lambda obj: dumps(obj).encode('utf-8'),
        lambda byte_obj: loads(byte_obj.decode('utf-8')),
        compress,
    )


def fix_path(
    path: PathLike, substitution_dict: Optional[Dict[str, str]] = None
) -> Path:
    path = ensure_resolved(path)
    path_str = str(path)

    if substitution_dict is None:
        substitution_dict = {
            ' ': '\\ ',
            '?': '\\?',
            '&': '\\&',
            '(': '\\(',
            ')': '\\)',
            '*': '\\*',
            '<': '\\<',
            '>': '\\>',
        }

    for orig, dest in substitution_dict.items():
        path_str = path_str.replace(orig, dest)

    return Path(path_str)


# ---------------------------------- Timer ----------------------------------
@contextmanager
def elapsed_timer():
    # Source: <https://stackoverflow.com/a/61613140/5811400>

    class _Timer:
        pass

    _Timer.start = default_timer()
    yield _Timer
    _Timer.end = default_timer()
    _Timer.duration = _Timer.end - _Timer.start


# ---------------------------------- Class printing ----------------------------------
def simple_pprint(self, obj, stream, indent, allowance, context, level):
    """
    Modified from pprint dict https://github.com/python/cpython/blob/3.7/Lib/pprint.py#L194
    """
    # Source: <https://stackoverflow.com/a/52521743/5811400>
    write = stream.write

    class_name = obj.__class__.__name__
    write(class_name + '(')
    _format_kwarg_dict_items(
        self,
        obj.__dict__.copy().items(),
        stream,
        indent + len(class_name),
        allowance + 1,
        context,
        level,
    )
    write(')')


def simple_pprinter(names: Optional[Tuple[str, ...]] = None):
    def simple_pprint(self, obj, stream, indent, allowance, context, level):
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
                values = empty_gen()
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

    return simple_pprint


def _format_kwarg_dict_items(
    self, items, stream, indent, allowance, context, level
):
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


def simple_pprint_class(cls: type, *names):
    pprint.PrettyPrinter._dispatch[cls.__repr__] = simple_pprinter(names)

    # Returning the class allows the use of this function as a decorator
    return cls


# ---------------------------------- Mixins ----------------------------------
class NamedMixin:
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name: str = name


class FrozenNamedMixin:
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name


class SimpleRepr:
    """A mixin implementing a simple __repr__."""

    # Source: <https://stackoverflow.com/a/44595303/5811400>

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        address = id(self) & 0xFFFFFF
        attrs = ', '.join(
            f'{key}={value!r}' for key, value in self.__dict__.items()
        )

        return f'<{class_name} @{address:x} {attrs}>'


class SimpleStr:
    def __str__(self) -> str:
        class_name = self.__class__.__name__
        attrs = ', '.join(
            f'{key}={value}' for key, value in self.__dict__.items()
        )

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


# ---------------------------------- Timing ----------------------------------


def get_timestamp(fmt='%Y-%m-%dT%H:%M:%SZ'):
    return datetime.datetime.now().strftime(fmt)


# ---------------------------------- Enum ----------------------------------
class NoValueEnum(enum.Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


def enum_item(
    enumeration: Type[_EnumType], item: Union[_EnumType, int, str]
) -> _EnumType:
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
