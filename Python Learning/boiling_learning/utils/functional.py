from dataclasses import dataclass
from functools import (
    partial,
    wraps
)
import itertools
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

from frozendict import frozendict
import more_itertools as mit


# TODO: when variadic generics are available, they will be very useful here


T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
ArgType = Sequence[T]
KwargType = Mapping[str, S]


@dataclass(frozen=True)
class Pack(Generic[T, S]):
    args: ArgType[T] = ()
    kwargs: KwargType[S] = frozendict()

    # @classmethod
    # def from_args(cls: Callable[[Sequence[T], Mapping[str, S]], U], *args: T, **kwargs: S) -> U:
    #     return cls(args, kwargs)

    def __iter__(self) -> Iterator[Union[ArgType[T], KwargType[S]]]:
        return (self.args, self.kwargs).__iter__()

    def as_pair(self) -> Tuple[tuple, dict]:
        return (
            tuple(self.args),
            dict(self.kwargs)
        )

    def feed(self, f: Callable[..., U]) -> U:
        return f(*self.args, **self.kwargs)

    def partial(self, f: Callable[..., U]) -> Callable[..., U]:
        return partial(
            f,
            *self.args,
            **self.kwargs
        )

    def rpartial(self, f: Callable[..., U]) -> Callable[..., U]:
        return rpartial(
            f,
            *self.args,
            **self.kwargs
        )


def pack(*args: T, **kwargs: S) -> Pack[T, S]:
    return Pack(args, kwargs)


def unpack(f: Callable[..., U], packed_param: Pack[T, S]) -> U:
    return packed(f)(packed_param)


def packed(f: Callable[..., U]) -> Callable[[Pack], U]:
    @wraps(f)
    def wrapper(pack: Pack) -> U:
        return pack.feed(f)
    return wrapper


def unpacked(f: Callable[[Pack[T, S]], U]) -> Callable[..., U]:
    @wraps(f)
    def wrapper(*args: T, **kwargs: S) -> U:
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
        arg: Iterable
) -> None:
    mit.consume(
        itertools.starmap(f, arg)
    )


def zip_filter(
        f: Callable[..., bool],
        *args: Iterable
) -> Iterator:
    yield from (
        tpl
        for tpl in zip(*args)
        if f(*tpl)
    )
