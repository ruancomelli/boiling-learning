from contextlib import suppress
from pathlib import Path
from typing import Callable, Generic, Iterable, Tuple, TypeVar, Union

from boiling_learning.io.io import LoaderFunction, PathLike, SaverFunction
from boiling_learning.utils.utils import ensure_parent

_T = TypeVar('_T')
CreatorFunction = Callable[[], _T]


class Persister(Generic[_T]):
    def __init__(
        self, saver: SaverFunction[_T], loader: LoaderFunction[_T]
    ) -> None:
        self.saver: SaverFunction[_T] = saver
        self.loader: LoaderFunction[_T] = loader

    def save(self, obj: _T, filepath: PathLike) -> None:
        self.saver(obj, filepath)

    def load(self, filepath: PathLike) -> _T:
        return self.loader(filepath)


class FileManager(Generic[_T]):
    def __init__(self, filepath: PathLike, persister: Persister[_T]) -> None:
        self.path: Path = ensure_parent(filepath)
        self.persister: Persister = persister

    def save(self, obj: _T) -> None:
        self.persister.save(obj, self.path)

    def load(self) -> _T:
        return self.persister.load(self.path)


class Provider(FileManager[_T]):
    def __init__(
        self,
        filepath: PathLike,
        persister: Persister[_T],
        creator: CreatorFunction[_T],
        exceptions: Union[Exception, Iterable[Exception]] = (),
    ) -> None:
        super().__init__(filepath, persister)

        if isinstance(exceptions, Exception):
            exceptions = (exceptions,)
        else:
            exceptions = tuple(exceptions)

        self.creator: CreatorFunction[_T] = creator
        self.exceptions: Tuple[Exception, ...] = exceptions

    def provide(self) -> _T:
        if self.path.exists():
            with suppress(*self.exceptions):
                return self.load()

        return self.creator()
