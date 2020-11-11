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
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)

from frozendict import frozendict
import funcy
import more_itertools as mit


# TODO: when variadic generics are available, they will be very useful here
T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
ArgsType = Sequence[T]
KwargsType = Mapping[str, S]


def nth_arg(n: int) -> Callable:
    def _nth(*args):
        args_len = len(args)
        if args_len > n:
            return args[n]
        else:
            raise TypeError(
                f'cannot get the {n}-th argument of a {args_len}-length tuple.'
                f' Got the following arguments: {args}'
            )
    return _nth


@dataclass
class Pack(Generic[T, S]):
    args: ArgsType[T] = ()
    kwargs: KwargsType[S] = frozendict()

    def __getitem__(self, idx_or_key: Union[int, str]) -> Union[T, S]:
        if isinstance(idx_or_key, int):
            return self.args[idx_or_key]
        elif isinstance(idx_or_key, str):
            return self.kwargs[idx_or_key]
        else:
            raise ValueError(
                f'*Pack* expects an *int* index or *str* key, but got a {type(idx_or_key)}')

    def __iter__(self) -> Iterator[Union[ArgsType[T], KwargsType[S]]]:
        return (self.args, self.kwargs).__iter__()

    def __str__(self) -> str:
        args2str = ', '.join(map(str, self.args))
        kwargs2str = ', '.join(
            f'{key}={value}'
            for key, value in self.kwargs.items()
        )
        return ''.join((
            'Pack(',
            ', '.join(
                ([args2str] if args2str else [])
                + ([kwargs2str] if kwargs2str else [])
            ),
            ')'
        ))

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

    def omit(
            self,
            pred: Callable[[T], bool],
            pos: Optional[Union[int, Iterable[int]]] = (),
            key: Optional[Union[str, Iterable[str]]] = ()
    ) -> 'Pack':
        '''
        p = pack(1, 2, None, 4, None, 6, a='a', b='b', c=None, d='d', e=None, f='f')
        print(p)
        p = p.omit(lambda x: x is None, pos=[0, 2], key=['d', 'e'])
        print(p)
        '''
        enumerated_args = tuple(enumerate(self.args))
        if pos is None:
            pos = funcy.walk(0, enumerated_args)
        elif isinstance(pos, int):
            pos = {pos}
        pos = frozenset(pos)
        to_remove = funcy.select_keys(
            pos,
            enumerated_args
        )
        to_remove = funcy.select_values(pred, to_remove)
        to_remove = frozenset(funcy.walk(0, to_remove))
        args = tuple(funcy.select_keys(
            lambda idx: idx not in to_remove,
            enumerated_args
        ))
        args = funcy.walk(1, args)

        if key is None:
            key = self.kwargs.keys()
        elif isinstance(key, str):
            key = (key,)
        key = tuple(key)
        to_remove = funcy.project(self.kwargs, key)
        to_remove = funcy.select_values(pred, to_remove)
        kwargs = funcy.omit(self.kwargs, to_remove.keys())

        return Pack(args=args, kwargs=kwargs)


def pack(*args: T, **kwargs: S) -> Pack[T, S]:
    return Pack(args, frozendict(kwargs))


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


def map_values(
        f: Callable[[T], S],
        iterable: Iterable[T]
) -> Iterable[S]:
    if hasattr(iterable, 'items'):
        return funcy.walk_values(f, iterable)
    else:
        return funcy.walk(f, iterable)


def zip_filter(
        f: Callable[..., bool],
        *args: Iterable
) -> Iterator:
    yield from (
        tpl
        for tpl in zip(*args)
        if f(*tpl)
    )
