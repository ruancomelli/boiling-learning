from typing import Any, Callable, Iterable, Iterator, Tuple, TypeVar

import more_itertools as mit

_T = TypeVar('_T')


def flaglast(iterable: Iterable[_T]) -> Iterator[Tuple[bool, _T]]:
    """Return pairs `(flag, element)` for each `element` in `iterable` where `flag` is `False` for all elements except the last one, for which it is `True`.

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
        Iterator[Tuple[bool, _T]]: Iterable of elements `(flag, elem)` for each `elem` in `iterable` with `flag` being `True` for the last element.
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

    This function is very similar to `list(map(side_effect, iterable))` or `[side_effect(element) for element in iterable]`, except that it is faster and does not store values. This is why the mapped function is called `side_effect`.

    Examples:
    ```py
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

    ```

    Args:
        side_effect (Callable[[_T], Any]): side effect to be applied to iterable. Should be a function accepting the elements yielded by the iterable.
        iterable (Iterable[_T]): iterable containing elements to be fed to the side effect function.
    """
    mit.consume(map(side_effect, iterable))
