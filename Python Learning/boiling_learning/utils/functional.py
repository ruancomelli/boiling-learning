import functools
from functools import (
    partial,
    reduce,
    wraps
)
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Tuple,
    TypeVar
)

import more_itertools as mit

T = TypeVar('T')
PackType = Tuple[tuple, Dict[str, Any]]


    def rpartial(self, f: Callable[..., U]) -> Callable[..., U]:
        return rpartial(
            f,
            *self.args,
            **self.kwargs
        )


def pack(*args, **kwargs):
    return args, kwargs


def unpack(f, packed_param):
    return packed(f)(packed_param)


def packed(f):
    @wraps(f)
    def wrapper(params):
        # params[0] == args
        # params[1] == kwargs
        return f(*params[0], **params[1])
    return wrapper


def unpacked(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(pack(*args, **kwargs))

    return wrapper


def reverse_args(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*reversed(args), **kwargs)
    return wrapper


def rpartial(f: Callable[..., U], *args, **kwargs) -> Callable[..., U]:
    # source: based on <https://github.com/Suor/funcy/pull/96/commits/0772ccac6803b143d8f54f365080f17a51633891>
    return lambda *a, **kw: f(*(a + args), **{**kwargs, **kw})


def apply(
        f: Callable[..., Any],
        *args
) -> None:
    mit.consume(
        map(f, *args)
    )


def starapply(
        f: Callable[..., Any],
        arg: Iterable) -> None:
    mit.consume(
        itertools.starmap(f, arg)
    )


def zip_filter(
        f: Callable[..., bool],
        *args: Iterable,
        star: bool = False
) -> Iterator:
    yield from (
        tpl
        for tpl in zip(*args)
        if f(*tpl)
    )
