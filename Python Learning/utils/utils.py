import matplotlib.pyplot as plt
import numpy as np

def empty(*args, **kwargs):
    pass

def fold(*args, **kwargs):
    return args, kwargs

def relative_path(origin, destination):
    from os.path import relpath
    return relpath(destination, start=origin)

def missing_elements(int_list): # source: adapted from <https://stackoverflow.com/questions/16974047/efficient-way-to-find-missing-elements-in-an-integer-sequence>
    if int_list:
        start, end = int_list[0], int_list[-1]
        full_list = set(range(start, end + 1))
        return sorted(full_list.difference(int_list))
    else:
        return set([])

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def dict_complement(dict_, keys):
    return {
        key: value
        for key, value in dict_.items()
        if key not in dict_
    }
    
def extract_value(dicts, key=None, keys=None, inverted=False, default_behaviour='value', default=None):    
    if isinstance(dicts, dict):
        dicts = [dicts]
        
    if not (None in (key, keys)):
        raise ValueError(f'either key or keys must be None')
    elif key is None:
        return {
            key: extract_value(dicts, key=key, default_behaviour='value', default=default)
            for key in keys
        }
    else:
        for d in dicts:
            if key in d:
                return d[key]
            
        default_behaviour = default_behaviour.lower()
            
        if default_behaviour == 'value':
            return default
        elif default_behaviour == 'raise':
            raise ValueError(f'key {key} was not found in the dictionaries.')
        else:
            raise ValueError(f'default_behaviour must be either \'value\' or \'raise\'. Got \'{default_behaviour}\'.')
        
def get_nested(dict_, *keys, **kwargs):
    available_kwargs = ['default']
    if not all(key in available_kwargs for key in kwargs):
        raise TypeError(f'invalid named arguments. Expected only keys in {available_kwargs}, got {list(kwargs.keys())}')
        
    if 'default' in kwargs:
        for key in keys:
            dict_ = dict_.get(key, kwargs['default'])
    else:
        for key in keys:
            dict_ = dict_[key]
    return dict_

def set_nested(dict_, *keys, value):
    for key in keys[:-1]:
        dict_ = dict_.setdefault(key, {})
    dict_[keys[-1]] = value

def print_value(
    name: str, value,
    *args,
    append_new_line: bool = False,
    fmt: str = '{name} = {value}',
    note: str = '',
    note_fmt: str = ' ({note})',
    **kwargs):
    """Standardized method for printing named values.

    Parameters
    ----------
    name            : name to be printed
    value           : value to be printed
    *args           : variadic arguments to be forwarded to the print function
    append_new_line : if True, prints a new line after everything was printed
    fmt             : format string to be used
    note            : note to be included after formatting
    note_fmt        : format for the note
    **kwargs        : variadic keyworded arguments to be forwarded to the print function
    """
    if note:
        fmt = fmt + note_fmt.format(note=note)
    print(fmt.format(name=name, value=value), *args, **kwargs)
        
    if append_new_line:
        print()
        
def print_array(
    name: str, value,
    *args,
    **kwargs):
    """Standardized method for printing named arrays.

    Parameters
    ----------
    name     : name to be printed
    value    : array to be printed
    *args    : variadic arguments to be forwarded to the print function
    **kwargs : variadic keyworded arguments to be forwarded to the print function
    """
    if value.ndim == 1:
        fmt = '{name} = {value}'
        append_new_line = False
    elif value.ndim == 2:
        fmt = '{name} =\n{value}'
        append_new_line = True
    else:
        raise ValueError('expecting a 1-D or 2-D array')
    print_value(
        name, value,
        *args,
        append_new_line=append_new_line,
        fmt=fmt,
        note='Shape: {shape}'.format(shape=value.shape),
        note_fmt=' ({note})',
        **kwargs)

def print_bool(
    name: str, value: bool,
    *args,
    append_new_line: bool = False,
    note: str = '',
    note_fmt: str = ' ({note})',
    **kwargs):
    """Standardized method for printing boolean named values.

    Parameters
    ----------
    name            : name to be printed
    value           : value to be printed
    *args           : variadic arguments to be forwarded to the print function
    append_new_line : if True, prints a new line after everything was printed
    note            : note to be included after formatting
    note_fmt        : format for the note
    **kwargs        : variadic keyworded arguments to be forwarded to the print function
    """
    print_value(
        name, value,
        *args,
        append_new_line=append_new_line,
        fmt='{name}: {value}',
        note=note,
        note_fmt=note_fmt,
        **kwargs)

def print_close(
    name_lhs: str, name_rhs: str,
    value_lhs, value_rhs,
    *args,
    tol: float = 1e-10,
    append_new_line: bool = False,
    note: str = '',
    note_fmt: str = ' ({note})',
    **kwargs):
    """Standardized method for printing boolean named values.

    Parameters
    ----------
    name            : name to be printed
    value           : value to be printed
    *args           : variadic arguments to be forwarded to the print function
    append_new_line : if True, prints a new line after everything was printed
    note            : note to be included after formatting
    note_fmt        : format for the note
    **kwargs        : variadic keyworded arguments to be forwarded to the print function
    """
    print_bool(
        f'{name_lhs} == {name_rhs}',
        np.all(np.abs(value_lhs - value_rhs) <= tol),
        *args,
        append_new_line=append_new_line,
        note=note,
        note_fmt=note_fmt,
        **kwargs)

def print_header(
    s: str,
    level: int = 0,
    levels=['=', '-', '~', '*']):
    """Standardized method for printing a section header.

    Prints the argument s underlined.
    
    Parameters
    ----------
    s       : string to be printed
    level   : index of level symbol to be used
    levels  : list of level symbols to choose from
    """
    s = str(s)
    print()
    print(s)
    print(levels[level] * len(s))
    
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
    
    import operator
                
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
    import json
    return json.loads( json.dumps(obj, cls=cls) )

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
        
    # idx = list_arg_where(groups, cond)
    # if idx is None:
    #     raise(ValueError(f'there is no support for {values_dict}'))
    # else:
    #     return idx

# ---------------------------------- Custom search ----------------------------------
def list_arg_where(lst, cond):
    for idx, value in enumerate(lst):
        if cond(value):
            return idx
    else:
        return None
