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


def identity(arg: T) -> T:
    return arg


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


def rpartial(f, *args, **kwargs):
    return partial(
        reverse_args(f),
        *args,
        **kwargs
    )


def apply(
        f: Callable[..., Any],
        *args) -> None:
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
