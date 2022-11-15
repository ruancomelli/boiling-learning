from contextlib import suppress
from typing import Callable, Generic, Iterable, Type, TypeVar

from loguru import logger

from boiling_learning.io import LoaderFunction, SaverFunction
from boiling_learning.utils.pathutils import PathLike, resolve

_T = TypeVar('_T')
CreatorFunction = Callable[[], _T]


class Persister(Generic[_T]):
    def __init__(self, saver: SaverFunction[_T], loader: LoaderFunction[_T]) -> None:
        self.saver = saver
        self.loader = loader

    def save(self, obj: _T, filepath: PathLike) -> None:
        resolved = resolve(filepath, parents=True)

        logger.debug(f'Saving result to {resolved}')
        self.saver(obj, resolved)
        logger.debug('Result saved')

    def load(self, filepath: PathLike) -> _T:
        resolved = resolve(filepath, parents=True)

        logger.debug(f'Loading result from {resolved}')
        result = self.loader(resolved)
        logger.debug('Result successfully loaded')

        return result


class Provider(Generic[_T]):
    def __init__(
        self,
        persister: Persister[_T],
        creator: CreatorFunction[_T],
        exceptions: Iterable[Type[Exception]] = (
            FileNotFoundError,
            NotADirectoryError,
        ),
        autosave: bool = True,
    ) -> None:
        self.persister = persister
        self.creator = creator
        self.exceptions = tuple(exceptions)
        self.autosave = autosave

    def provide(self, filepath: PathLike) -> _T:
        logger.debug(f'Providing result for file {filepath}')

        resolved = resolve(filepath)

        if resolved.exists():
            with suppress(*self.exceptions):
                return self.persister.load(resolved)

        logger.debug('Unable to load result, creating...')
        obj = self.creator()
        logger.debug('Result created')

        if self.autosave:
            self.persister.save(obj, resolved)

        return obj
