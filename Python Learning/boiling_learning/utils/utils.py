import re
from pathlib import Path
from collections import ChainMap
import operator
from itertools import product

import matplotlib.pyplot as plt
import more_itertools as mit
from more_itertools import unzip

_sentinel = object()

def empty(*args, **kwargs):
    pass

def constant(value):
    def wrapper(*args, **kwargs):
        return value
    return wrapper

def constant_callable(callable_value):
    def wrapper(*args, **kwargs):
        return callable_value()
    return wrapper

def comment(f, s: str = ''):
    from functools import wraps
    
    @wraps(f)
    def wrapped(*args, **kwargs):
        if s:
            print(s)
        else:
            print(f)
        return f(*args, **kwargs)
    return wrapped

def remove_duplicates(iterable, key=None):
    # See <https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.unique_everseen>
    
    if key is None:
        return mit.unique_everseen(iterable)
    elif callable(key):    
        return mit.unique_everseen(
            iterable,
            key=key
        )
    elif isinstance(key, dict):            
        return mit.unique_everseen(
            iterable,
            key=lambda elem: key[type(elem)]
        )
    elif key == 'fast':
        return remove_duplicates(
            iterable,
            key={
                list: tuple,
                set: frozenset,
                dict: (lambda elem: frozenset(elem.items())),
            }
        )

def has_duplicates(iterable):
    try:
        iterable_len = len(iterable)
    except TypeError:
        iterable_len = len(list(iterable))
    return iterable_len != len(set(iterable))

def missing_elements(int_list):
    # source: adapted from <https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence>
    if int_list:
        int_list = sorted(int_list)
        start, end = int_list[0], int_list[-1]
        full = range(start, end + 1)
        return list(filter(lambda x: x not in int_list, full))
    else:
        return []

def merge_dicts(*dict_args, latter_precedence=True):
    if latter_precedence:
        dict_args = reversed(dict_args)
    
    return dict(ChainMap(*dict_args))

def projection(*indices):
    def wrapped(*args):
        return tuple(args[i] for i in indices)
    return wrapped

def partial_isinstance(type_):
    def wrapped(x):
        return isinstance(x, type_)
    return wrapped

def alternate_iter(
    iterables,
    default_indices=None,
    skip_repeated=True
):    
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
    if default_indices is None:
        default_indices = (0,) * len(iterables)
    else:
        default_indices = tuple(default_indices)
        default_indices = default_indices + (0,)*(len(iterables) - len(default_indices))
        
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
                iterables[iterable_index+1:], default_indices[iterable_index+1:]
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

def combine_dict(dct, gen=None):
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

def dict_complement(dict_, keys):
    return {
        key: value
        for key, value in dict_.items()
        if key not in keys
    }
    
def dict_product(**kwargs):
    # source: <https://stackoverflow.com/a/5228294/5811400>
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def extract_value(dicts, key=None, default=_sentinel, many=False):
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
                raise ValueError(f'key {key} was not found in the dictionaries.')
        else:
            return cm.get(key, default)

def invert_dict(d):
    return dict((v, k) for k, v in d.items())

def extract_keys(d, value, cmp=operator.eq):
    for k, v in d.items():
        if cmp(value, v):
            yield k
        
# ---------------------------------- Operators ----------------------------------
def is_None(x):
    return x is None
def is_not_None(x):
    return x is not None

# ---------------------------------- Printing ----------------------------------
def print_verbose(verbose, *args, **kwargs):
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
    s = str(s)
    print_verbose(verbose)
    print_verbose(verbose, s)
    print_verbose(verbose, levels[level] * len(s))
    
# ---------------------------------- Plotting functions ----------------------------------
def prepare_fig(
    n_cols: int = None,
    n_rows: int = None,
    n_elems: int = None,
    fig_size=None,
    subfig_size=None,
    tight_layout: bool = True):
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
        raise ValueError('incompatible arguments: either figsize or subfig_size must be None')
    if None not in (n_cols, n_rows, n_elems):
        raise ValueError('incompatible arguments: either n_cols, n_rows or n_elems must be None')

    grid_size = None
    if None not in (n_cols, n_rows):
        grid_size = (n_rows, n_cols)
    elif None not in (n_cols, n_elems):
        n_rows = (n_elems-1)//n_cols + 1
        grid_size = (n_rows, n_cols)
    elif None not in (n_rows, n_elems):
        n_cols = (n_elems-1)//n_rows + 1
        grid_size = (n_rows, n_cols)
    
    def validate(size):
        if size in ['micro']:
            return (2, 1.5)
        if size in ['tiny']:
            return (4, 3)
        if size in ['small']:
            return (7, 5)
        elif size in ['normal', 'intermediate']:
            return (9, 7)
        elif size in ['large', 'big']:
            return (18, 15)
        else:
            return size
        
    fig_size = validate(fig_size)
    subfig_size = validate(subfig_size)
                
    if subfig_size:
        fig_size = (grid_size[1] * subfig_size[0], grid_size[0] * subfig_size[1])
    
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

def as_json(obj, cls=None):
    from json import loads, dumps
    return loads(dumps(obj, cls=cls))

def json_equivalent(lhs, rhs, cls=None):
    return as_json(lhs, cls) == as_json(rhs, cls)

# ---------------------------------- Default parameters ----------------------------------
def regularize_default(
    x, cond, default,
    many=False, many_cond=False, many_default=False,
    call_default=False
):
    # Manage default arguments and passed parameters
    from itertools import repeat

    if many:
        cond = cond if many_cond else repeat(cond)
        default = default if many_default else repeat(default)

        return (
            regularize_default(
                x_, cond_, default_,
                many=False, many_cond=False, many_default=False,
                call_default=call_default
            )
            for x_, cond_, default_ in zip(x, cond, default)
        )

    if cond(x):
        return x
    else:
        if call_default:
            return default()
        else:
            return default

def check_value_match(groups, values_dict):    
    def cond(value):
        return all(value[1][key](v) for key, v in values_dict.items())
    
    try:
        return next(
            filter(cond, enumerate(groups))
        )[0]
    except StopIteration:
        raise ValueError(f'there is no support for {values_dict}')

# ---------------------------------- Array operations ----------------------------------
def crop_array(
    a,
    lims=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    x_lim=None,
    y_lim=None
):
    idx = check_value_match(
        [
            dict(
                lims=is_not_None,
                x_min=is_None,
                x_max=is_None,
                y_min=is_None,
                y_max=is_None,
                x_lim=is_None,
                y_lim=is_None
            ),
            dict(
                lims=is_None,
                x_min=is_not_None,
                x_max=is_not_None,
                y_min=is_not_None,
                y_max=is_not_None,
                x_lim=is_None,
                y_lim=is_None
            ),
            dict(
                lims=is_None,
                x_min=is_None,
                x_max=is_None,
                y_min=is_None,
                y_max=is_None,
                x_lim=is_not_None,
                y_lim=is_not_None
            )
        ],
        dict(
            lims=lims,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_lim=x_lim,
            y_lim=y_lim
        )
    )
    if idx == 0:
        return a[tuple(slice(*lim) for lim in lims)]
    elif idx == 1:
        return crop_array(a, lims=[(x_min, x_max), (y_min, y_max)])
    elif idx == 2:
        return crop_array(a, lims=[x_lim, y_lim])
    
def shift_array(
    a,
    shifts=None,
    shift_x=None,
    shift_y=None
):
    idx = check_value_match(
        [
            {
                'shifts': is_None,
                'shift_x': is_not_None,
                'shift_y': is_not_None,
            },
            {
                'shifts': is_not_None,
                'shift_x': is_None,
                'shift_y': is_None,
            }
        ],
        {
            'shifts': shifts,
            'shift_x': shift_x,
            'shift_y': shift_y
        }
    )
    if idx == 0:
        return shift(a, shifts=(shift_x, shift_y))
    
    def _shift(a, shifts, axis=0):
        from numpy import roll
        
        if shifts:
            return _shift(
                roll(a, shifts[0], axis=axis),
                shifts[1:],
                axis+1
            )
        else:
            return a
    return _shift(a, shifts, axis=0)

# ---------------------------------- Iteration ----------------------------------
def append(iterable, value):
    from itertools import chain
    
    return chain(iterable, [value])

# ---------------------------------- Path ----------------------------------
def relative_path(origin, destination):
    from os.path import relpath
    return relpath(destination, start=origin)

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

# Source: https://stackoverflow.com/a/57892171/5811400
def rmdir(path, recursive=False, keep=False, missing_ok=False):
    path = Path(path)
    
    if not path.is_dir():
        if missing_ok:
            return
        else:
            raise NotADirectoryError(f'path is expected to be a directory when missing_ok={missing_ok}. Got {path}.')
    
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
              
def group_files(path, keyfunc=None):
    if keyfunc is None:
        keyfunc = lambda x: x.suffix

    d = {}
    for p in filter(lambda x: x.is_file(), path.iterdir()):
        d.setdefault(keyfunc(p), []).append(p)
    return d      
        
def mover(
    out_dir,
    up_level: int = 0,
    make_dir: bool = True,
    head_aggregator=None,
    verbose: bool = False
):
    out_dir = Path(out_dir)
    
    def wrapper(in_path):
        in_path = Path(in_path)
        head = in_path.parents[up_level]
        tail = in_path.relative_to(head)
        
        if head_aggregator is None:
            out_path = out_dir / tail
        else:
            out_path = out_dir.with_name(head_aggregator.join([head.name, out_dir.name])) / tail
        
        if make_dir:
            out_path.parent.mkdir(exist_ok=True, parents=True)
            
        print_verbose(verbose, f'mover: up_level={up_level}; make_dir={make_dir}; head_aggregator={head_aggregator}')
        print_verbose(verbose, in_path)
        print_verbose(verbose, '>', out_path)
        
        return out_path
        
    return wrapper

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
    return list(f(dir_path) for f in fs), {
        path.name: dir_as_tree_apply(path, fs, dir_pred=dir_pred)
        for path in dir_path.iterdir()
        if path.is_dir() and (dir_pred is None or dir_pred(path))
    }

# ---------------------------------- Timer ----------------------------------
from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def elapsed_timer():
    # Source: <https://stackoverflow.com/a/61613140/5811400>
    
    start_time = default_timer()

    class _Timer:
        @property
        def start(self):
            return start_time
        
        @property
        def end(self):
            return default_timer()
        
        @property
        def duration(self):
            return self.end - self.start

    yield _Timer

    end_time = default_timer()
    _Timer.end = end_time
    _Timer.duration = end_time - start_time
    
# ---------------------------------- Class printing ----------------------------------
class SimpleRepr:
    """A mixin implementing a simple __repr__."""
    # Source: <https://stackoverflow.com/a/44595303/5811400>
    def __repr__(self):
        class_name = self.__class__.__name__
        address = id(self) & 0xFFFFFF
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())
        
        return f'<{class_name} @{address:x} {attrs}>'
        
class SimpleStr:
    def __str__(self):
        class_name = self.__class__.__name__
        attrs = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        
        return f'{class_name}({attrs})'
        
def simple_pprint(self, obj, stream, indent, allowance, context, level):
    """
    Modified from pprint dict https://github.com/python/cpython/blob/3.7/Lib/pprint.py#L194
    """
    # Source: <https://stackoverflow.com/a/52521743/5811400>
    write = stream.write
    
    cls = obj.__class__
    write(cls.__name__ + "(")
    _format_kwarg_dict_items(
        self, obj.__dict__.copy().items(), stream, indent + len(cls.__name__), allowance + 1, context, level
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
        self._format(ent, stream, indent + len(key) + 1,
                        allowance if last else 1,
                        context, level)
        if not last:
            write(delimnl)