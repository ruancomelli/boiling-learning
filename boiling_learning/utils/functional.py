from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from itertools import chain
from typing import (
    Any,
    Generic,
    TypeVar,
    overload,
)

import funcy
from frozendict import frozendict  # type: ignore[attr-defined]

from boiling_learning.descriptions import describe

# TODO: when variadic generics are available, they will be very useful here
_T = TypeVar("_T")
_S = TypeVar("_S")
_U = TypeVar("_U")


class Pack(Generic[_T, _S]):
    def __init__(
        self, args: Iterable[_T] = (), kwargs: Mapping[str, _S] = frozendict()
    ) -> None:
        self._args = tuple(args)
        self._kwargs = frozendict(kwargs)

    @property
    def args(self) -> tuple[_T, ...]:
        return self._args

    @property
    def kwargs(self) -> frozendict[str, _S]:
        return self._kwargs

    def pair(self) -> tuple[tuple[_T, ...], frozendict[str, _S]]:
        return (self._args, self._kwargs)

    def __bool__(self) -> bool:
        return bool(self.args) or bool(self.kwargs)

    def __eq__(self, other: Any) -> bool:
        return self is other or (
            isinstance(other, Pack)
            and self.args == other.args
            and self.kwargs == other.kwargs
        )

    def __hash__(self) -> int:
        return hash(self.pair())

    def __getitem__(self, loc: int | str) -> _T | _S:
        return self.args[loc] if isinstance(loc, int) else self.kwargs[loc]

    def __iter__(self) -> Iterator[tuple[_T] | frozendict[str, _S]]:
        return iter(self.pair())

    def __repr__(self) -> str:
        return f"Pack(args={self.args}, kwargs={self.kwargs})"

    def __str__(self) -> str:
        arguments = chain(
            map(str, self.args),
            (f"{key}={value}" for key, value in self.kwargs.items()),
        )
        return f"P({', '.join(arguments)})"

    def __describe__(self) -> tuple[tuple[_T, ...], frozendict[str, _S]]:
        return describe(self.pair())  # type: ignore[arg-type]

    def feed(self, f: Callable[..., _U]) -> _U:
        return f(*self.args, **self.kwargs)

    def partial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return partial(f, *self.args, **self.kwargs)

    def rpartial(self, f: Callable[..., _U]) -> Callable[..., _U]:
        return funcy.rpartial(f, *self.args, **self.kwargs)

    @overload
    def __matmul__(self, other: Callable[..., _U]) -> Callable[..., _U]: ...

    @overload
    def __matmul__(self, other: Any) -> Any: ...

    def __matmul__(self, other: Any) -> Any:
        return self.partial(other) if callable(other) else NotImplemented

    @overload
    def __rmatmul__(self, other: Callable[..., _U]) -> Callable[..., _U]: ...

    @overload
    def __rmatmul__(self, other: Any) -> Any: ...

    def __rmatmul__(self, other: Any) -> Any:
        return self.rpartial(other) if callable(other) else NotImplemented


class P(Pack[_T, _S], Generic[_T, _S]):
    def __init__(self, *args: _T, **kwargs: _S) -> None:
        super().__init__(args, kwargs)
