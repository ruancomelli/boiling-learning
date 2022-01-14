from pathlib import Path
from typing import Any, Callable, Generic, Iterable, TypeVar, Union

from typing_extensions import ParamSpec

from boiling_learning.io.io import LoaderFunction, SaverFunction, load_json, save_json
from boiling_learning.management.allocators.json_allocator import default_table_allocator
from boiling_learning.management.persister import FileProvider, Provider
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike

_P = ParamSpec('_P')
_R = TypeVar('_R')
_CallableT = TypeVar('_CallableT', bound=Callable[..., Any])


class Cacher(Generic[_R]):
    def __init__(
        self,
        allocator: Callable[[Pack], Path],
        saver: SaverFunction[_R],
        loader: LoaderFunction[_R],
        exceptions: Union[Exception, Iterable[Exception]] = (
            FileNotFoundError,
            NotADirectoryError,
        ),
        autosave: bool = True,
    ) -> None:
        self.allocator: Callable[[Pack], Path] = allocator
        self.saver: SaverFunction[_R] = saver
        self.loader: LoaderFunction[_R] = loader
        self.exceptions: Union[Exception, Iterable[Exception]] = exceptions
        self.autosave: bool = autosave


class CachedFunction(Generic[_P, _R]):
    def __init__(self, function: Callable[_P, _R], cacher: Cacher[_R]) -> None:
        self.function: Callable[_P, _R] = function
        self.cacher: Cacher[_R] = cacher

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        path: Path = self.allocate(*args, **kwargs)
        creator: Callable[[], _R] = Pack(args, kwargs).partial(self.function)

        provider: FileProvider = FileProvider(
            path,
            Provider(
                saver=self.cacher.saver,
                loader=self.cacher.loader,
                creator=creator,
                exceptions=self.cacher.exceptions,
                autosave=self.cacher.autosave,
            ),
        )

        return provider.provide()

    def allocate(self, *args: _P.args, **kwargs: _P.kwargs) -> Path:
        return self.cacher.allocator(Pack(args, kwargs))


def function_cacher(cacher: Cacher[_R]) -> Callable[[Callable[_P, _R]], CachedFunction[_P, _R]]:
    def _wrapper(function: Callable[_P, _R]) -> CachedFunction[_P, _R]:
        return CachedFunction(function, cacher)

    return _wrapper


def cache(
    allocator: Callable[[Pack], Path],
    saver: SaverFunction[_R],
    loader: LoaderFunction[_R],
    exceptions: Union[Exception, Iterable[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> Callable[[Callable[_P, _R]], CachedFunction[_P, _R]]:
    return function_cacher(
        Cacher(
            allocator=allocator,
            saver=saver,
            loader=loader,
            exceptions=exceptions,
            autosave=autosave,
        )
    )


def json_cache(root: PathLike, autosave: bool = True) -> Callable[[_CallableT], _CallableT]:
    return cache(
        allocator=default_table_allocator(root),
        saver=save_json,
        loader=load_json,
        autosave=autosave,
    )
