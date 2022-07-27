from datetime import timedelta
from timeit import default_timer
from types import TracebackType
from typing import Optional, Type


class Timer:
    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> 'Timer':
        self._start = default_timer()
        self._end = None
        return self

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> None:
        self._end = default_timer()

    @property
    def duration(self) -> Optional[timedelta]:
        if self._start is None:
            return None

        if self._end is None:
            return timedelta(seconds=default_timer() - self._start)

        return timedelta(seconds=self._end - self._start)
