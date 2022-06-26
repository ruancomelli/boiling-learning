from typing import Any, Callable, Iterable, Iterator, List, Tuple, TypeVar

import more_itertools as mit
import numpy as np
from typing_extensions import Literal

_T = TypeVar('_T')


def flaglast(iterable: Iterable[_T]) -> Iterator[Tuple[bool, _T]]:
    """Return pairs `(flag, element)` for each `element` in `iterable` where `flag` is
    `False` for all elements except the last one, for which it is `True`.

    Example:
    >>> flagged = list(flaglast(range(4)))
    >>> flagged
    [(False, 0), (False, 1), (False, 2), (True, 3)]
    >>> for is_last, elem in flaglast('RUAN'):
    ...     if is_last:
    ...         print(f'Last letter: "{elem}"')
    Last letter: "N"

    Args:
        iterable (Iterable[_T]): Iterable whose last element we want to be flagged.

    Yields:
        Iterator[Tuple[bool, _T]]: Iterable of elements `(flag, elem)` for each `elem`
        in `iterable` with `flag` being `True` for the last element.
    """
    it = iter(iterable)

    try:
        last = next(it)
    except StopIteration:
        return

    for val in it:
        yield False, last
        last = val

    yield True, last


def apply(side_effect: Callable[[_T], Any], iterable: Iterable[_T]) -> None:
    """Apply side effect to elements in iterable.

    This function is very similar to `list(map(side_effect, iterable))` or
    `[side_effect(element) for element in iterable]`, except that it is faster and does
    not store values. This is why the mapped function is called `side_effect`.

    Examples:

    >>> numbers = [1, 2, 3]
    >>> apply(print, numbers)
    1
    2
    3
    >>> source = 'abc'
    >>> dest = []
    >>> apply(dest.append, source)
    >>> dest
    ['a', 'b', 'c']

    Args:
        side_effect (Callable[[_T], Any]): side effect to be applied to iterable.
        Should be a function accepting the elements yielded by the iterable.
        iterable (Iterable[_T]): iterable containing elements to be fed to the side
        effect function.
    """
    mit.consume(map(side_effect, iterable))


def evenly_spaced_indices(
    total: int, count: int, *, goal: Literal['distance', 'spread']
) -> List[int]:
    if not 0 <= count <= total:
        raise ValueError('`total` and `count` must satisfy the constraint `0 <= count <= total`.')

    if count == 0:
        return []

    if count == 1:
        return [total // 2]

    if count == total:
        return list(range(total))

    if goal == 'distance':
        # maximizing distance means:
        # 1 0 0 0 0 0 0 1
        points = np.linspace(0, total - 1, count)
    elif goal == 'spread':
        # maximizing spread means:
        # 0 0 1 0 0 1 0 0
        points = np.linspace(0, total, count + 2)[1:-1]
    else:
        raise ValueError('`goal` must be either `"distance"` or `"spread"`')

    return np.round(points).astype(int).tolist()


def distance_maximized_evenly_spaced_indices(total: int, count: int) -> List[int]:
    return evenly_spaced_indices(total, count, goal='distance')
