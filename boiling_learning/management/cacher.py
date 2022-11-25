from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, ParamSpec, Type, TypeVar

from loguru import logger

from boiling_learning.io import LoaderFunction, SaverFunction
from boiling_learning.io.storage import load, save
from boiling_learning.management.allocators import Allocator
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.pathutils import PathLike, resolve

_P = ParamSpec('_P')
_R = TypeVar('_R')
CreatorFunction = Callable[[], _R]


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

    def provide(self, creator: CreatorFunction[_R], path: Path) -> _R:
        logger.debug(f'Providing result for file {path}')

        resolved = resolve(path)

        if resolved.exists():
            with suppress(*self.exceptions):
                logger.debug(f'Loading result from {resolved}')
                result = self.loader(resolved)
                logger.debug('Result successfully loaded')

                return result

        logger.debug('Unable to load result, creating...')
        obj = creator()
        logger.debug('Result created')

        if self.autosave:
            logger.debug(f'Saving result to {resolved}')
            self.saver(obj, resolve(resolved, parents=True))
            logger.debug('Result saved')

        return obj

    def allocate(self, *args: Any, **kwargs: Any) -> Path:
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
        creator: CreatorFunction[_R] = self.function @ Pack(args, kwargs)

        return self.provide(creator, path)

    def load(self, path: PathLike) -> _R:
        return self.cacher.load(path)

    def save(self, obj: _R, path: PathLike) -> None:
        self.cacher.save(obj, path)

    def provide(self, creator: CreatorFunction[_R], path: Path) -> _R:
        return self.cacher.provide(creator, path)

    def allocate(self, *args: _P.args, **kwargs: _P.kwargs) -> Path:
        return self.cacher.allocate(*args, **kwargs)


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
    cacher = Cacher(
        allocator=allocator,
        saver=saver,
        loader=loader,
        exceptions=exceptions,
        autosave=autosave,
    )

    def decorator(function: Callable[_P, _R]) -> CachedFunction[_P, _R]:
        return CachedFunction(function, cacher)

    return decorator
