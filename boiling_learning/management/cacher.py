from functools import partial
from pathlib import Path
from typing import Callable, Iterable, TypeVar, Union

import wrapt

from boiling_learning.io.io import LoaderFunction, SaverFunction
from boiling_learning.management.persister import FileProvider, Provider
from boiling_learning.utils.functional import Pack

_T = TypeVar('_T')
_Callable = TypeVar('_Callable', bound=Callable)


def cache(
    allocator: Callable[[Pack], Path],
    saver: SaverFunction[_T],
    loader: LoaderFunction[_T],
    exceptions: Union[Exception, Iterable[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> Callable[[_Callable], _Callable]:
    @wrapt.decorator
    def cacher(
        wrapped: Callable[..., _T], instance: None, args: tuple, kwargs: dict
    ) -> _T:
        pack: Pack = Pack(args, kwargs)
        path: Path = allocator(pack)

        provider: FileProvider = FileProvider(
            path,
            Provider(
                saver=saver,
                loader=loader,
                creator=partial(wrapped, *args, **kwargs),
                exceptions=exceptions,
                autosave=autosave,
            ),
        )

        return provider.provide()

    return cacher
