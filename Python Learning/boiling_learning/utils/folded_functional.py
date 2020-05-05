import boiling_learning.utils
# TODO: from folded to packed
class Folded:
    def __init__(self, f):
        self._fun = f
        
    def __call__(self, args, kwargs):
        return self._fun(*args, **kwargs)
    
def List(*args):
    return (
		args,
		{}
    )

def Dict(**kwargs):
    return (
		[],
		kwargs
	)
    
def compose(*ffs, filter_None=False):
    from functools import reduce
    
    def _compose2(f, g):
        if filter_None:
            if f is None:
                f = boiling_learning.utils.identity
            if g is None:
                g = boiling_learning.utils.identity
        
        def result(x):
            return g(f(x))
        return result
    
    return reduce(_compose2, ffs, boiling_learning.utils.identity)