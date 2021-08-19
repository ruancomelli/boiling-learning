from __future__ import annotations

import itertools
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import funcy
import more_itertools as mit

from boiling_learning.utils.FrozenDict import FrozenDict

# TODO: when variadic generics are available, they will be very useful here
_T = TypeVar('_T')
_S = TypeVar('_S')
_U = TypeVar('_U')
ArgsType = Sequence[_T]
KwargsType = Mapping[str, _S]


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


class Pack(Hashable, Generic[_T, _S]):
    def __init__(
        self, args: ArgsType[_T] = (), kwargs: KwargsType[_S] = FrozenDict()
    ) -> None:
        self._args: Tuple[_T] = tuple(args)
        self._kwargs: FrozenDict[str, _S] = FrozenDict(kwargs)

    @property
    def args(self) -> Tuple[_T]:
        return self._args

    @property
    def kwargs(self) -> KwargsType[_S]:
        return self._kwargs

    def __bool__(self) -> bool:
        return bool(self.args) or bool(self.kwargs)

    def __eq__(self, other: Pack) -> bool:
        return self is other or (
            isinstance(other, self.__class__)
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash((self.args, self.kwargs))

    def __getitem__(self, loc: Union[int, str]) -> Union[_T, _S]:
        if isinstance(loc, int):
            return self.args[loc]
        elif isinstance(loc, str):
            return self.kwargs[loc]
        else:
            raise ValueError(
                f'*Pack* expects an *int* index or *str* key, but got a {type(loc)}'
            )

    def __iter__(self) -> Iterator[Union[Tuple[_T], KwargsType[_S]]]:
        return iter((self.args, self.kwargs))

    def __repr__(self) -> str:
        return f'Pack(args={self.args}, kwargs={self.kwargs})'

    def __str__(self) -> str:
        args2str = ', '.join(map(str, self.args))
        kwargs2str = ', '.join(
            f'{key}={value}' for key, value in self.kwargs.items()
        )
        return ''.join(
            (
                'P(',
                args2str,
                ', ' if args2str and kwargs2str else '',
                kwargs2str,
                ')',
            )
        )

    def __json_encode__(self) -> Dict[str, Union[list, dict]]:
        return {'args': list(self.args), 'kwargs': dict(self.kwargs)}

    def __json_decode__(self, **data) -> None:
        '''Decode JSON object as Pack.

        Expects the object to be in the format *{"args": args, "kwargs": kwargs]*, e.g.:
        {
            "my_pack": {
                "args": [1, 2, "name"],
                "kwargs": {
                    "number": 3.14,
                    "phone": null
                }
            }
        }
        '''
        self._args = data['args']
        self._kwargs = data['kwargs']

    @classmethod
    def pack(*cls_n_args: _T, **kwargs: _S) -> Pack[_T, _S]:
        cls, *args = cls_n_args
        return cls(args, kwargs)

    def feed(self, f: Callable[..., _U]) -> _U:
        return f(*self.args, **self.kwargs)

    def partial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return partial(f, *self.args, **self.kwargs)

    def rpartial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return funcy.rpartial(f, *self.args, **self.kwargs)

    def omit(
        self,
        loc: Union[
            int, str, Iterable[Union[int, str]]
        ] = (),  # TODO: unify everything here
        pred: Optional[Callable[[_T], bool]] = None,
    ) -> Pack:
        '''
        p = P(1, 2, None, 4, None, 6, a='a', b='b', c=None, d='d', e=None, f='f')
        p2 = p.omit((0, 2, 'd', 'e'))
        print(p2) # prints Pack(2, 4, None, 6, a=a, b=b, c=None, f=f)
        p3 = p.omit((0, 2, 'd', 'e'), lambda x: x is None)
        print(p3) # prints Pack(1, 2, 4, None, 6, a=a, b=b, c=None, d=d, f=f)
        '''
        if isinstance(loc, int):
            pos = frozenset({loc})
            key = ()
        elif isinstance(loc, str):
            pos = frozenset()
            key = (loc,)
        else:
            pos = frozenset(filter(funcy.isa(int), loc))
            key = tuple(filter(funcy.isa(str), loc))

        enumerated_args = tuple(enumerate(self.args))
        to_remove = funcy.select_keys(pos, enumerated_args)
        if pred is not None:
            to_remove = funcy.select_values(pred, to_remove)
        to_remove = frozenset(funcy.walk(0, to_remove))
        args = tuple(
            funcy.select_keys(
                lambda idx: idx not in to_remove, enumerated_args
            )
        )
        args = funcy.walk(1, args)

        to_remove = funcy.project(self.kwargs, key)
        if pred is not None:
            to_remove = funcy.select_values(pred, to_remove)
        kwargs = funcy.omit(self.kwargs, to_remove.keys())

        return Pack(args=args, kwargs=kwargs)

    def _copy(self, new_args, new_kwargs, right: bool = False) -> Pack:
        n_new_args = len(new_args)
        if right:
            args = self.args[:-n_new_args] + new_args
        else:
            args = new_args + self.args[n_new_args:]
        kwargs = self.kwargs.union(new_kwargs)
        return Pack(args, kwargs)

    def copy(self, *new_args, **new_kwargs) -> Pack:
        '''
        p1 = P(1, 'a', 'Hi', x=0, y='hello')
        p2 = p1.copy(0, 'b', x='bye', z='byello')
        print(p2) # prints Pack(0, b, Hi, x=bye, y=hello, z=byello)
        '''
        return self._copy(new_args, new_kwargs, right=False)

    def rcopy(self, *new_args, **new_kwargs) -> Pack:
        '''
        p1 = P(1, 'a', 'Hi', x=0, y='hello')
        p2 = p1.rcopy(0, 'b', x='bye', z='byello')
        print(p2) # prints Pack(1, 0, b, x=bye, y=hello, z=byello)
        '''
        return self._copy(new_args, new_kwargs, right=True)

    def _apply(self, fargs, fkwargs, right: bool = False) -> Pack:
        n_fargs = len(fargs)

        args_to_transform = (
            self.args[-n_fargs:] if right else self.args[:n_fargs]
        )

        new_args = tuple(
            f(arg) if f is not None else arg
            for f, arg in zip(fargs, args_to_transform)
        )
        new_kwargs = {k: f(self[k]) for k, f in fkwargs.items()}

        return self._copy(new_args, new_kwargs, right=right)

    def apply(self, *fargs, **fkwargs) -> Pack:
        '''
        p1 = P(1, 'a', 'Hi', x=0, y='hello')
        p2 = p1.apply(lambda value: value+5, x=len, y=str.upper)
        print(p2) # prints Pack(5, b, Hi, x=3, y=HELLO, z=byello)
        '''
        return self._apply(fargs, fkwargs, right=False)

    def rapply(self, *fargs, **fkwargs) -> Pack:
        '''
        p1 = P(1, 'a', 'Hi', x=0, y='hello')
        p2 = p1.rapply(lambda value: value+5, x=len, y=str.upper)
        print(p2) # prints Pack(0, b, hi, x=3, y=HELLO, z=byello)
        '''
        return self._apply(fargs, fkwargs, right=True)


P = Pack.pack


def unpack(f: Callable[..., _U], packed_param: Pack[_T, _S]) -> _U:
    return packed(f)(packed_param)


def pack_combinations(
    combinator: Callable[..., Iterator], pack: Pack[Sequence, Sequence]
) -> Iterator[Pack]:
    keys = tuple(pack.kwargs.keys())
    values = tuple(pack.kwargs.values())
    n_args = len(pack.args)

    for combination in combinator(*pack.args, *values):
        yield Pack(
            combination[:n_args], funcy.zipdict(keys, combination[n_args:])
        )


def packed(f: Callable[..., _U]) -> Callable[[Pack], _U]:
    @wraps(f)
    def wrapper(pack: Pack) -> _U:
        return pack.feed(f)

    return wrapper


def unpacked(f: Callable[[Pack[_T, _S]], _U]) -> Callable[..., _U]:
    @wraps(f)
    def wrapper(*args: _T, **kwargs: _S) -> _U:
        return f(Pack(args, kwargs))

    return wrapper


def reverse_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*reversed(args), **kwargs)

    return wrapper


def apply(f: Callable[..., Any], *args) -> None:
    mit.consume(map(f, *args))


def starapply(f: Callable[..., Any], arg: Iterable) -> None:
    mit.consume(itertools.starmap(f, arg))


def map_values(f: Callable[[_T], _S], iterable: Iterable[_T]) -> Iterable[_S]:
    if hasattr(iterable, 'items'):
        return funcy.walk_values(f, iterable)
    else:
        return funcy.walk(f, iterable)


def zip_filter(f: Callable[..., bool], *args: Iterable) -> Iterator:
    yield from (tpl for tpl in zip(*args) if f(*tpl))
