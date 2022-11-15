from contextlib import suppress
from typing import Callable, Iterable, Type, TypeVar

from loguru import logger

from boiling_learning.io import LoaderFunction, SaverFunction
from boiling_learning.utils.pathutils import PathLike, resolve

_T = TypeVar('_T')
CreatorFunction = Callable[[], _T]


def provide(
    filepath: PathLike,
    /,
    *,
    saver: SaverFunction[_T],
    loader: LoaderFunction[_T],
    creator: CreatorFunction[_T],
    exceptions: Iterable[Type[Exception]] = (
        FileNotFoundError,
        NotADirectoryError,
    ),
    autosave: bool = True,
) -> _T:
    logger.debug(f'Providing result for file {filepath}')

    resolved = resolve(filepath)

    if resolved.exists():
        with suppress(*exceptions):
            logger.debug(f'Loading result from {resolved}')
            result = loader(resolved)
            logger.debug('Result successfully loaded')

            return result

    logger.debug('Unable to load result, creating...')
    obj = creator()
    logger.debug('Result created')

    if autosave:
        logger.debug(f'Saving result to {resolved}')
        saver(obj, resolve(filepath, parents=True))
        logger.debug('Result saved')

    return obj
