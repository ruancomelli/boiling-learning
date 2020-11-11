import os
import re
from pathlib import Path
from collections import (
    ChainMap,
    defaultdict
)
from collections.abc import (
    MutableSet
)
import datetime
import enum
from itertools import product
from functools import (
    wraps,
    partial
)
from timeit import default_timer
from typing import (
    Any,
    Callable,
    Collection,
    Container,
    Dict,
    Hashable,
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
    overload
)
import shutil
import tempfile
from contextlib import contextmanager
import pprint
import enum
import dataclasses
from dataclasses import dataclass

import funcy
import matplotlib.pyplot as plt
import more_itertools as mit
from more_itertools import unzip
from pandas.api.types import union_categoricals
import modin.pandas as pd
from sortedcontainers import SortedSet


# ---------------------------------- Typing ----------------------------------
class _SentinelType(enum.Enum):
    SENTINEL = enum.auto()


_sentinel = _SentinelType.SENTINEL
T = TypeVar('T')
SentinelOptional = Union[_SentinelType, T]


# see <https://www.python.org/dev/peps/pep-0519/#provide-specific-type-hinting-support>
PathType = Union[str, 'os.PathLike[str]']
VerboseType = Union[bool, int]
JSONDataType = Union[None, bool, int, float, str, List['JSONDataType'], Dict[str, 'JSONDataType']]


# ---------------------------------- Utility functions ----------------------------------
def empty(*args, **kwargs) -> None:
    pass


@overload
def indexify(arg: Iterable) -> Iterable[int]: ...


@overload
def indexify(arg: Collection) -> range: ...


def indexify(arg):
    try:
        return range(len(arg))
    except TypeError:
        return map(
            operator.itemgetter(0),
            enumerate(arg)
        )


# TODO: in Python 3.7+, use Literal[True] and Literal[False] to distinguish between the possible input and output types
def constant(value: T) -> Callable[..., T]:
    def _constant(*args, **kwargs):
        return value
    return _constant


def constant_factory(value: Callable[[], T]) -> Callable[..., T]:
    def _constant(*args, **kwargs):
        return value()
    return _constant


def comment(
    f: Callable,
    s: str = '',
    printer: Callable[[str], Any] = print
) -> Callable:
    @wraps(f)
    def wrapped(*args, **kwargs):
        printer(s)
        return f(*args, **kwargs)
    return wrapped


def remove_duplicates(
    iterable: Iterable[T],
    key: Union[None, object, Dict, Callable] = _sentinel
) -> Iterable[T]:
    # See <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_everseen>

    if key is _sentinel:
        # use default optimization
        return remove_duplicates(
            iterable,
            key={
                list: tuple,
                set: frozenset,
                dict: (lambda elem: frozenset(elem.items())),
            }
        )
    elif isinstance(key, dict):
        return mit.unique_everseen(
            iterable,
            key=lambda elem: key.get(type(elem), funcy.identity)(elem)
        )
    else:
        return mit.unique_everseen(
            iterable,
            key=key
        )


def reorder(
    seq: Sequence[T],
    indices: Iterable[int]
) -> Iterable[T]:
    return map(seq.__getitem__, indices)


def argmin(
    iterable: Iterable
) -> int:
    return min(
        enumerate(iterable),
        key=operator.itemgetter(1)
    )[0]


def argmax(
    iterable: Iterable
) -> int:
    return max(
        enumerate(iterable),
        key=operator.itemgetter(1)
    )[0]


def argsorted(iterable: Iterable) -> Iterable[int]:
    return map(
        operator.itemgetter(0),
        sorted(
            enumerate(iterable),
            key=operator.itemgetter(1)
        )
    )


def multipop(
    lst: List,
    indices: Container[int]
):
    pop = [lst[i] for i in indices]
    lst[:] = [v for i, v in enumerate(lst) if i not in indices]

    return pop


def has_duplicates(iterable: Iterable) -> bool:
    if isinstance(iterable, list):
        iterable_list = iterable
    elif isinstance(iterable, set):
        return False
    else:
        iterable_list = list(iterable)

    return len(iterable_list) != mit.ilen(remove_duplicates(iterable_list))


def missing_ints(ints: Iterable[int]) -> Iterable[int]:
    # source: adapted from <https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence>
    ints = SortedSet(ints)
    if ints:
        start, end = ints[0], ints[-1]
        full = range(start, end + 1)
        return itertools.filterfalse(ints.__contains__, full)
    else:
        return empty_gen()


def is_consecutive(
        ints: Iterable[int],
        ignore_order: bool = False
) -> bool:
    ints = tuple(ints)
    if not ints:
        return True

    if ignore_order:
        # Source: https://stackoverflow.com/a/64177833/5811400
        r = range(min(ints), max(ints) + 1)
        return (
            len(ints) == len(r)
            and all(map(ints.__contains__, r))
        )
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


def alternate_iter(
    iterables: Iterable[Iterable],
    default_indices: Iterable[int] = tuple(),
    skip_repeated: bool = True
) -> Iterator[tuple]:
    '''
        >>> alternate_iter(
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
    default_indices = default_indices + \
        (0,)*(len(iterables) - len(default_indices))

    if skip_repeated:
        yield tuple(iterable[default_indices[idx]] for idx, iterable in enumerate(iterables))

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
                iterables[iterable_index +
                          1:], default_indices[iterable_index+1:]
            )
        )
        if skip_repeated:
            for idx, item in enumerate(iterable):
                if idx == default_indices[iterable_index]:
                    continue
                else:
                    yield head + (item,) + tail
        else:
            for item in iterable:
                yield head + (item,) + tail


def combine_dict(
    dct: Mapping[Any, Iterable],
    gen: Optional[Callable[[list], Iterable[Iterable]]] = None
) -> Iterator[dict]:
    if gen is None:
        def default_gen(x):
            return product(*x)
        gen = default_gen

    keys, iterables = unzip(dct.items())
    keys = list(keys)
    iterables = list(iterables)

    for combination in gen(iterables):
        yield dict(
            zip(keys, combination)
        )


def dict_complement(
    dict_: Mapping,
    keys: Container
) -> dict:
    return {
        key: value
        for key, value in dict_.items()
        if key not in keys
    }


def dict_product(**kwargs: Mapping) -> Iterator[dict]:
    # source: <https://stackoverflow.com/a/5228294/5811400>
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


def extract_value(
        dicts,
        key=None,
        default=_sentinel,
        many=False
):
    if many:
        return {
            k: extract_value(dicts, key=k, default=default, many=False)
            for k in key
        }
    else:
        cm = ChainMap(*dicts)
        if default is _sentinel:
            if key in cm:
                return cm[key]
            else:
                raise ValueError(
                    f'key {key} was not found in the dictionaries.')
        else:
            return cm.get(key, default)


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
    return {
        v: dct[k]
        for k, v in key_map.items()
    }


class KeyedDefaultDict(defaultdict):
    '''

    Source: https://stackoverflow.com/a/2912455/5811400
    '''

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def is_dataclass_instance(obj) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_dataclass_class(Type) -> bool:
    return dataclasses.is_dataclass(Type) and isinstance(Type, type)


def dataclass_from_mapping(
        mapping: Mapping[str, Any],
        dataclass_factory: Callable[..., T],
        key_map: Optional[Union[dataclass, Mapping[str, str]]] = None
) -> T:
    if not is_dataclass_class(dataclass_factory):
        raise ValueError('*dataclass_factory* must be a dataclass.')

    dataclass_field_names = set(map(
        operator.attrgetter('name'),
        dataclasses.fields(dataclass_factory)
    ))

    if key_map is None:
        return dataclass_factory(
            **funcy.select_keys(
                dataclass_field_names,
                mapping
            )
        )
    else:
        if is_dataclass_instance(key_map):
            key_map = dataclasses.asdict(key_map)

        key_map = funcy.select_keys(
            dataclass_field_names,
            key_map
        )
        translator = invert_dict(key_map).get
        mapping = {
            translator(key, key): value
            for key, value in mapping.items()
        }
        return dataclass_from_mapping(
            mapping,
            dataclass_factory
        )


def to_parent_dataclass(obj: dataclass, parent_factory: Callable[..., T]) -> T:
    if not is_dataclass_class(parent_factory):
        raise ValueError('*parent_factory* must be a dataclass.')

    if not is_dataclass_instance(obj):
        raise ValueError('*obj* must be a dataclass instance.')

    if not isinstance(obj, parent_factory):
        raise ValueError('*obj* must be an instance of *parent_factory*.')

    return dataclass_from_mapping(
        dataclasses.asdict(obj),
        parent_factory
    )


def concatenate_dataframes(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate while preserving categorical columns.

    Source: <https://stackoverflow.com/a/57809778/5811400>

    NB: We change the categories in-place for the input dataframes"""

    dfs = tuple(dfs)

    # Iterate on categorical columns common to all dfs
    for col in set.intersection(*(
            set(df.select_dtypes(include='category').columns)
            for df in dfs
    )):
        # Generate the union category across dfs for this column
        uc = union_categoricals(tuple(df[col] for df in dfs))
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
    levels=('#', '=', '-', '~', '^', '*', '+'),
    verbose: bool = True
):
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
        tight_layout: bool = True
):
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
    if None not in (fig_size, subfig_size):
        raise ValueError(
            'incompatible arguments: either figsize or subfig_size must be None')
    if None not in (n_cols, n_rows, n_elems):
        raise ValueError(
            'incompatible arguments: either n_cols, n_rows or n_elems must be None')

    grid_size = None
    if None not in {n_cols, n_rows}:
        grid_size = (n_rows, n_cols)
    elif None not in {n_cols, n_elems}:
        n_rows = (n_elems-1)//n_cols + 1
        grid_size = (n_rows, n_cols)
    elif None not in {n_rows, n_elems}:
        n_cols = (n_elems-1)//n_rows + 1
        grid_size = (n_rows, n_cols)

    def validate(size):
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

    fig_size = validate(fig_size)
    subfig_size = validate(subfig_size)

    if subfig_size:
        fig_size = (grid_size[1] * subfig_size[0],
                    grid_size[0] * subfig_size[1])

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


def as_json(obj, cls: Optional[type] = None):
    from json import loads, dumps
    return loads(dumps(obj, cls=cls))


def json_equivalent(lhs, rhs, cls: Optional[type] = None) -> bool:
    return as_json(lhs, cls) == as_json(rhs, cls)


# ---------------------------------- Iteration ----------------------------------
def empty_gen():
    # Source: <https://stackoverflow.com/a/13243870/5811400>
    return
    yield


def append(iterable, value):
    yield from iterable
    yield value


def transpose(iterable):
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
    return append(
        drop_last(iterable),
        last_value
    )


# ---------------------------------- Path ----------------------------------
def relative_path(origin, destination):
    from os.path import relpath
    return relpath(destination, start=origin)


def ensure_resolved(
        path: PathType,
        root: Optional[PathType] = None
) -> Path:
    path = Path(path)
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def ensure_dir(
        path: PathType,
        root: Optional[PathType] = None
) -> Path:
    path = ensure_resolved(path, root=root)
    path.mkdir(exist_ok=True, parents=True)
    return path


def ensure_parent(
        path: PathType,
        root: Optional[PathType] = None
) -> Path:
    path = ensure_resolved(path, root=root)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def remove_copy(directory, pattern):
    def remove_copy_idx(path):
        pattern = re.compile('(.*) \([0-9]+\)\.png')
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
def is_parent_dir(parent: PathType, subdir: PathType) -> bool:
    parent = ensure_resolved(parent)
    subdir = ensure_resolved(subdir)
    return parent in subdir.parents


# Source: https://stackoverflow.com/a/57892171/5811400
def rmdir(path, recursive=False, keep=False, missing_ok=False):
    path = ensure_resolved(path)

    if not path.is_dir():
        if missing_ok:
            return
        else:
            raise NotADirectoryError(
                f'path is expected to be a directory when missing_ok={missing_ok}. Got {path}')

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


def mover(
    out_dir,
    head: Optional[PathType] = None,
    up_level: int = 0,
    make_dir: bool = True,
    head_aggregator=None,
    verbose: VerboseType = False
):
    if make_dir:
        out_dir = ensure_dir(out_dir)
    else:
        out_dir = ensure_resolved(out_dir)

    if head is not None and up_level != 0:
        raise ValueError('when \'head\' is given, up_level must be 0')

    if head is not None:
        def get_head(in_path):
            return head
    else:
        # up_level is not None
        def get_head(in_path):
            return in_path.parents[up_level]

    def wrapper(in_path: PathType):
        in_path = ensure_resolved(in_path)
        head = get_head(in_path)
        tail = in_path.relative_to(head)

        if head_aggregator is None:
            out_path = out_dir / tail
        else:
            out_path = out_dir.with_name(
                head_aggregator.join([head.name, out_dir.name])) / tail

        if make_dir:
            out_path = ensure_parent(out_path)

        if verbose >= 2:
            print(f'{in_path} -> {out_path}')

        return out_path

    return wrapper


def dir_as_tree(dir_path, file_pred=None, dir_pred=None):
    ret_list = []
    ret_dict = {}

    for path in dir_path.iterdir():
        if path.is_file() and (file_pred is None or file_pred(path)):
            ret_list.append(path)
        elif path.is_dir() and (dir_pred is None or dir_pred(path)):
            ret_dict[path.name] = dir_as_tree(
                path, file_pred=file_pred, dir_pred=dir_pred)

    return ret_list, ret_dict


def dir_as_tree_apply(dir_path, fs, dir_pred=None):
    return list(f(dir_path) for f in fs), {
        path.name: dir_as_tree_apply(path, fs, dir_pred=dir_pred)
        for path in dir_path.iterdir()
        if path.is_dir() and (dir_pred is None or dir_pred(path))
    }


def count_file_lines(path):
    with open(path) as f:
        return mit.ilen(f)


@contextmanager
def tempdir(suffix=None, prefix=None, dir=None):
    dirpath = Path(
        tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
    ).resolve().absolute()

    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


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
    write(class_name + "(")
    _format_kwarg_dict_items(
        self, obj.__dict__.copy().items(), stream, indent + len(class_name), allowance + 1, context, level
    )
    write(")")


def _format_kwarg_dict_items(self, items, stream, indent, allowance, context, level):
    '''
    Modified from pprint dict https://github.com/python/cpython/blob/3.7/Lib/pprint.py#L194
    '''
    write = stream.write

    indent += self._indent_per_level
    delimnl = ',\n' + ' ' * indent
    last_index = len(items) - 1
    for i, (key, ent) in enumerate(items):
        last = i == last_index
        write(key)
        write('=')
        self._format(
            ent, stream, indent + len(key) + 1,
            allowance if last else 1,
            context, level
        )
        if not last:
            write(delimnl)


def simple_pprint_class(cls: type):
    pprint.PrettyPrinter._dispatch[cls.__repr__] = simple_pprint

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
        attrs = ', '.join(f'{key}={value!r}' for key,
                          value in self.__dict__.items())

        return f'<{class_name} @{address:x} {attrs}>'


class SimpleStr:
    def __str__(self) -> str:
        class_name = self.__class__.__name__
        attrs = ', '.join(f'{key}={value}' for key,
                          value in self.__dict__.items())

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


# ---------------------------------- Collections ----------------------------------
class Ranges(MutableSet):
    @staticmethod
    def to_range(start, stop=None):
        if stop is None:
            return range(start, start+1)
        else:
            step = 1 if stop >= start else -1
            return range(start, stop+1, step)

    def __init__(self, ints: Iterable[int]):
        self.ranges = SortedSet(
            key=operator.itemgetter(0)
        )

        # Source: https://stackoverflow.com/a/47642650/5811400
        for group in mit.consecutive_groups(sorted(ints)):
            group = list(group)
            if len(group) == 1:
                self.ranges.add(self.to_range(group[0]))
            else:
                self.ranges.add(self.to_range(group[0], group[-1]))

        self._len = sum(map(len, self.ranges))

    def __contains__(self, elem):
        return any(
            map(contains(elem), self.ranges)
        )

    def __iter__(self):
        return itertools.chain.from_iterable(self.ranges)

    def __len__(self):
        return self._len

    def _merge_ranges(self, idx):
        self.ranges[idx].stop += len(self.ranges[idx+1])
        self.remove(idx+1)

    def add(self, elem):
        if elem not in self:
            relem = self.to_range(elem)
            self.ranges.add(relem)
            idx = self.ranges.index(relem)

            if idx < len(self.ranges)-1 and elem == self.ranges[idx+1][0]-1:
                self._merge_ranges(idx)

            if idx > 0 and elem == self.ranges[idx-1].stop:
                self._merge_ranges(idx-1)

    def discard(self, elem):
        for idx, r in enumerate(self.ranges):
            if elem in r:
                break
        else:
            raise KeyError(elem)

        if elem == r[0]:
            self.ranges[idx].start += 1
        elif elem == r[-1]:
            self.ranges[idx].stop -= 1
        else:
            del self.ranges[idx]
            self.ranges.add(self.to_range(r[0], elem))
            self.ranges.add(self.to_range(elem+1, r[-1]))

    def __str__(self):
        as_str = ', '.join(
            (f'[{r[0]} {r[-1]}]' if len(r) > 1 else str(r[0]))
            for r in self.ranges
        )
        return f'Ranges({as_str})'


# class DateTimeRange(collections.abc.Sequence):
#     def __init__(
#         self,
#         start,
#         stop,
#         step
#     ):


# ---------------------------------- Timing ----------------------------------
def get_timestamp(fmt='%Y-%m-%dT%H:%M:%SZ'):
    return datetime.datetime.now().strftime(fmt)


# ---------------------------------- Enum ----------------------------------
class NoValueEnum(enum.Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)


# ---------------------------------- Argument generator ----------------------------------
class ArgGenerator:
    def __init__(
            self,
            generator: Callable[[Hashable], 'Pack'],
            keep: bool = False,
    ):
        self._keep = keep
        if keep:
            self._store = KeyedDefaultDict(generator)
            self._call = self._store.__getitem__
        else:
            self._call = generator

    def __call__(self, key: Hashable):
        return self._call(key)


class AutoGenerator:
    def __init__(
            self,
            generator: Callable[[Hashable], 'Pack'],
            keep: bool = False,
            key_index: int = 0
    ):
        self._arg_gen = ArgGenerator(generator, keep)
        self._key_index = key_index

    def __call__(self, f: Callable):
        @wraps(f)
        def wrapped(*args, **kwargs):
            args = list(args)
            key = args.pop(self._key_index)
            return self._arg_gen(key)(*args, **kwargs)
        return wrapped


# ---------------------------------- Operator ----------------------------------
def contains(elem):
    '''Return a predicated that tests if a container contains elem.'''
    return funcy.rpartial(operator.contains, elem)


def is_(x) -> Callable[[Any], bool]:
    return partial(operator.is_, x)


def is_not(x) -> Callable[[Any], bool]:
    return partial(operator.is_not, x)
