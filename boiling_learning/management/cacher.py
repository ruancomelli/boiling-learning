from __future__ import annotations

from pathlib import Path
from typing import Callable, Generic, Iterable, Type, TypeVar

from loguru import logger
from typing_extensions import Concatenate, ParamSpec

from boiling_learning.io import LoaderFunction, SaverFunction
from boiling_learning.io.storage import load, save
from boiling_learning.management.allocators import Allocator
from boiling_learning.management.persister import FileProvider, Provider
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike

# pylint: disable=missing-function-docstring,missing-class-docstring

_P = ParamSpec('_P')
_R = TypeVar('_R')


class Cacher(Generic[_R]):
    def __init__(
        self,
        allocator: Allocator,
        saver: SaverFunction[_R] = save,
        loader: LoaderFunction[_R] = load,
        exceptions: Iterable[Type[Exception]] = (
            FileNotFoundError,
            NotADirectoryError,
        ),
        autosave: bool = True,
    ) -> None:
        self.allocator = allocator
        self.saver = saver
        self.loader = loader
        self.exceptions = tuple(exceptions)
        self.autosave = autosave

    def provide(self, creator: Callable[[], _R], path: Path) -> _R:
        provider = FileProvider(
            path,
            Provider(
                saver=self.save,
                loader=self.load,
                creator=creator,
                exceptions=self.exceptions,
                autosave=self.autosave,
            ),
        )

        return provider.provide()

    def decorate(self, function: Callable[_P, _R]) -> CachedFunction[_P, _R]:
        return CachedFunction(function, self)

    def allocate(self, *args: _P.args, **kwargs: _P.kwargs) -> Path:
        return self.allocator.allocate(*args, **kwargs)

    def load(self, path: PathLike) -> _R:
        return self.loader(path)

    def save(self, obj: _R, path: PathLike) -> None:
        self.saver(obj, path)


class CachedFunction(Generic[_P, _R]):
    def __init__(self, function: Callable[_P, _R], cacher: Cacher[_R]) -> None:
        self.function = function
        self.cacher = cacher

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        path = self.allocate(*args, **kwargs)
        creator: Callable[[], _R] = Pack(args, kwargs).partial(self.function)

        return self.provide(creator, path)

    def load(self, path: PathLike) -> _R:
        return self.cacher.load(path)

    def save(self, obj: _R, path: PathLike) -> None:
        self.cacher.save(obj, path)

    def provide(self, creator: Callable[[], _R], path: Path) -> _R:
        return self.cacher.provide(creator, path)

    def allocate(self, *args: _P.args, **kwargs: _P.kwargs) -> Path:
        logger.debug(f'Allocating path for args {(args, kwargs)}')

        path = self.cacher.allocate(*args, **kwargs)

        logger.debug(f'Allocated path is {path}')

        return path


def cache(
    allocator: Allocator,
    saver: SaverFunction[_R] = save,
    loader: LoaderFunction[_R] = load,
    exceptions: Iterable[Type[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> Callable[[Callable[_P, _R]], CachedFunction[_P, _R]]:
    return Cacher(
        allocator=allocator,
        saver=saver,
        loader=loader,
        exceptions=exceptions,
        autosave=autosave,
    ).decorate


def cacher_aware(
    allocator: Allocator,
    saver: SaverFunction[_R] = save,
    loader: LoaderFunction[_R] = load,
    exceptions: Iterable[Type[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> Callable[[Callable[Concatenate[Cacher[_R], _P], _R]], CachedFunction[_P, _R]]:
    cacher = Cacher(
        allocator=allocator,
        saver=saver,
        loader=loader,
        exceptions=exceptions,
        autosave=autosave,
    )

    def _wrapper(wrapped: Callable[Concatenate[Cacher[_R], _P], _R]) -> CachedFunction[_P, _R]:
        def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return wrapped(cacher, *args, **kwargs)

        return cacher.decorate(_wrapped)

    return _wrapper
