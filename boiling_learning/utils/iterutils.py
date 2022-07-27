import itertools
from typing import Any, Callable, Iterable, Tuple, TypeVar

import more_itertools as mit

_T = TypeVar('_T')


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


def unsort(iterable: Iterable[_T]) -> Tuple[Iterable[int], Iterable[_T]]:
    peekable = mit.peekable(iterable)

    if not peekable:
        return (), ()

    sorted_indices, sorters = mit.sort_together((peekable, itertools.count()))
    _, unsorters = mit.sort_together((sorters, itertools.count()))
    return unsorters, sorted_indices
