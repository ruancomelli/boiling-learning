from contextlib import suppress
from pathlib import Path
from typing import Callable, Generic, Iterable, Tuple, TypeVar, Union

from boiling_learning.io.io import LoaderFunction, PathLike, SaverFunction
from boiling_learning.utils import ensure_parent, resolve

_T = TypeVar('_T')
CreatorFunction = Callable[[], _T]


class Persister(Generic[_T]):
    def __init__(self, saver: SaverFunction[_T], loader: LoaderFunction[_T]) -> None:
        self.saver: SaverFunction[_T] = saver
        self.loader: LoaderFunction[_T] = loader

    def save(self, obj: _T, filepath: PathLike) -> None:
        self.saver(obj, resolve(filepath))

    def load(self, filepath: PathLike) -> _T:
        return self.loader(resolve(filepath))


class FilePersister(Generic[_T]):
    def __init__(self, filepath: PathLike, persister: Persister[_T]) -> None:
        self.path: Path = ensure_parent(filepath)
        self.persister: Persister = persister

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
        exceptions: Union[Exception, Iterable[Exception]] = (
            FileNotFoundError,
            NotADirectoryError,
        ),
        autosave: bool = True,
    ) -> None:
        super().__init__(saver, loader)

        exceptions = (exceptions,) if isinstance(exceptions, Exception) else tuple(exceptions)

        self.creator: CreatorFunction[_T] = creator
        self.exceptions: Tuple[Exception, ...] = exceptions
        self.autosave: bool = autosave

    def provide(self, filepath: PathLike) -> _T:
        resolved: Path = resolve(filepath)

        if resolved.exists():
            with suppress(*self.exceptions):
                return self.load(resolved)

        obj: _T = self.creator()

        if self.autosave:
            self.save(obj, resolved)

        return obj


class FileProvider(FilePersister[_T]):
    def __init__(self, filepath: PathLike, provider: Provider[_T]) -> None:
        super().__init__(filepath, provider)

        self.provider: Provider[_T] = provider

    def provide(self) -> _T:
        return self.provider.provide(self.path)
