def apply_to_obj(tpl):
    if len(tpl) != 4:
        raise ValueError('expected a tuple in the format (obj, fname, args, kwargs)')
    
    obj, fname, args, kwargs = tpl
    return getattr(obj, fname)(*args, **kwargs)

def apply_to_f(tpl):
    if len(tpl) != 3:
        raise ValueError('expected a tuple in the format (f, args, kwargs)')
    
    f, args, kwargs = tpl
    return f(*args, **kwargs)

