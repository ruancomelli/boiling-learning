import functools
from functools import (
    partial
)


def identity(arg):
    return arg

def pack(*args, **kwargs):
    return args, kwargs

def unpack(f, packed_param):
    return packed(f)(packed_param)

def packed(f):
    from functools import wraps
    
    @wraps(f)
    def wrapper(params):
        # params[0] == args
        # params[1] == kwargs
        return f(*params[0], **params[1])
    return wrapper

def unpacked(f):    
    from functools import wraps
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(pack(*args, **kwargs))
    
    return wrapper    
    
def compose(*fs, filter_None=False):
    from functools import reduce
    
    def _compose(f, g):
        if filter_None:
            if f is None:
                f = identity
            if g is None:
                g = identity
        
        def result(x):
            return g(f(x))
        
        return result
    
    return reduce(_compose, fs, identity)

def reverse_args(f):
    from functools import wraps
    
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*reversed(args), **kwargs)
    return wrapper

def rpartial(f, *args, **kwargs):
    return partial(
        reverse_args(f),
        *args,
        **kwargs
    )
