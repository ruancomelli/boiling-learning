from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, TypeVar, Union

import wrapt

from boiling_learning.io.io import LoaderFunction, SaverFunction, load_json, save_json
from boiling_learning.management.allocators import default_table_allocator
from boiling_learning.management.persister import FileProvider, Provider
from boiling_learning.utils.functional import Pack
from boiling_learning.utils.utils import PathLike

_T = TypeVar('_T')
_ArgType = TypeVar('_ArgType')
_KwargType = TypeVar('_KwargType')
_CallableT = TypeVar('_CallableT', bound=Callable[..., Any])


def cache(
    allocator: Callable[[Pack], Path],
    saver: SaverFunction[_T],
    loader: LoaderFunction[_T],
    exceptions: Union[Exception, Iterable[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> Callable[[_CallableT], _CallableT]:
    @wrapt.decorator
    def cacher(
        wrapped: Callable[..., _T],
        instance: None,
        args: Tuple[_ArgType],
        kwargs: Dict[str, _KwargType],
    ) -> _T:
        pack: Pack[_ArgType, _KwargType] = Pack(args, kwargs)
        path: Path = allocator(pack)
        creator: Callable[[], _T] = pack.partial(wrapped)

        provider: FileProvider = FileProvider(
            path,
            Provider(
                saver=saver,
                loader=loader,
                creator=creator,
                exceptions=exceptions,
                autosave=autosave,
            ),
        )

        return provider.provide()

    return cacher


def json_cache(root: PathLike, autosave: bool = True) -> Callable[[_CallableT], _CallableT]:
    return cache(
        allocator=default_table_allocator(root),
        saver=save_json,
        loader=load_json,
        autosave=autosave,
    )
