def identity(arg):
    return arg

def pack(*args, **kwargs):
    return args, kwargs

def packed(f):
    from functools import wraps
    
    @wraps(f)
    def wrapper(params):
        # params[0] == args
        # params[1] == kwargs
        return f(*params[0], **params[1])
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
