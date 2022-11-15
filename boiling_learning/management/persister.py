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
        self.saver(obj, resolve(filepath))

    def load(self, filepath: PathLike) -> _T:
        return self.loader(resolve(filepath))


class FilePersister(Generic[_T]):
    def __init__(self, filepath: PathLike, persister: Persister[_T]) -> None:
        self.path = resolve(filepath, parents=True)
        self.persister = persister

    def save(self, obj: _T) -> None:
        self.persister.save(obj, self.path)

    def load(self) -> _T:
        return self.persister.load(self.path)


class Provider(Persister[_T]):
    def __init__(
        self,
        saver: SaverFunction[_T],
        loader: LoaderFunction[_T],
        creator: CreatorFunction[_T],
        exceptions: Iterable[Type[Exception]] = (
            FileNotFoundError,
            NotADirectoryError,
        ),
        autosave: bool = True,
    ) -> None:
        super().__init__(saver, loader)

        self.creator = creator
        self.exceptions = tuple(exceptions)
        self.autosave = autosave

    def provide(self, filepath: PathLike) -> _T:
        logger.debug(f'Providing result for file {filepath}')

        resolved = resolve(filepath)

        if resolved.exists():
            with suppress(*self.exceptions):
                result = self.load(resolved)
                logger.debug('Result successfully loaded')
                return result

        logger.debug('Unable to load result, creating...')

        obj = self.creator()

        logger.debug('Result created')

        if self.autosave:
            logger.debug(f'Saving result to {resolved}')
            self.save(obj, resolve(resolved, parents=True))
            logger.debug('Result saved')

        return obj
