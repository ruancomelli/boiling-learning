from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, TypeVar, Union, overload

import funcy
from frozendict import frozendict

from boiling_learning.descriptions import describe

# TODO: when variadic generics are available, they will be very useful here
_T = TypeVar('_T')
_S = TypeVar('_S')
_U = TypeVar('_U')


ArgsType = tuple[_T, ...]
KwargsType = frozendict[str, _S]


class Pack(Generic[_T, _S]):
    def __init__(self, args: Iterable[_T] = (), kwargs: Mapping[str, _S] = frozendict()) -> None:
        self._args = tuple(args)
        self._kwargs = frozendict(kwargs)

    @property
    def args(self) -> ArgsType[_T]:
        return self._args

    @property
    def kwargs(self) -> KwargsType[_S]:
        return self._kwargs

    def pair(self) -> tuple[ArgsType[_T], KwargsType[_S]]:
        return (self._args, self._kwargs)

    def __bool__(self) -> bool:
        return bool(self.args) or bool(self.kwargs)

    def __eq__(self, other: Any) -> bool:
        return self is other or (
            isinstance(other, Pack) and self.args == other.args and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash(self.pair())

    def __getitem__(self, loc: Union[int, str]) -> Union[_T, _S]:
        return self.args[loc] if isinstance(loc, int) else self.kwargs[loc]

    def __iter__(self) -> Iterator[Union[tuple[_T], KwargsType[_S]]]:
        return iter(self.pair())

    def __repr__(self) -> str:
        return f'Pack(args={self.args}, kwargs={self.kwargs})'

    def __str__(self) -> str:
        arguments = chain(
            map(str, self.args),
            (f'{key}={value}' for key, value in self.kwargs.items()),
        )
        return f'P({", ".join(arguments)})'

    def __describe__(self) -> tuple[tuple[_T], KwargsType[_S]]:
        return describe(self.pair())

    def feed(self, f: Callable[..., _U]) -> _U:
        return f(*self.args, **self.kwargs)

    def partial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return partial(f, *self.args, **self.kwargs)

    def rpartial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return funcy.rpartial(f, *self.args, **self.kwargs)

    @overload
    def __matmul__(self, other: Callable[..., _U]) -> Callable[..., _U]:
        ...

    @overload
    def __matmul__(self, other: Any) -> Any:
        ...

    def __matmul__(self, other: Any) -> Any:
        return self.partial(other) if callable(other) else NotImplemented

    @overload
    def __rmatmul__(self, other: Callable[..., _U]) -> Callable[..., _U]:
        ...

    @overload
    def __rmatmul__(self, other: Any) -> Any:
        ...

    def __rmatmul__(self, other: Any) -> Any:
        return self.rpartial(other) if callable(other) else NotImplemented

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

    def _copy(self, new_args, new_kwargs, right: bool = False) -> Pack:
        n_new_args = len(new_args)
        if right:
            args = self.args[:-n_new_args] + new_args
        else:
            args = new_args + self.args[n_new_args:]
        kwargs = frozendict({**self.kwargs, **new_kwargs})
        return Pack(args, kwargs)

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

    def _apply(self, fargs, fkwargs, right: bool = False) -> Pack:
        n_fargs = len(fargs)

        args_to_transform = self.args[-n_fargs:] if right else self.args[:n_fargs]

        new_args = tuple(
            f(arg) if f is not None else arg for f, arg in zip(fargs, args_to_transform)
        )
        new_kwargs = {k: f(self[k]) for k, f in fkwargs.items()}

        return self._copy(new_args, new_kwargs, right=right)


class P(Pack[_T, _S], Generic[_T, _S]):
    def __init__(self, *args: _T, **kwargs: _S) -> None:
        super().__init__(args, kwargs)
