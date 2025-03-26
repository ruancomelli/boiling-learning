from datetime import timedelta
from timeit import default_timer
from types import TracebackType


class Timer:
    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> "Timer":
        self._start = default_timer()
        self._end = None
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._end = default_timer()

    @property
    def duration(self) -> timedelta | None:
        if self._start is None:
            return None

        if self._end is None:
            return timedelta(seconds=default_timer() - self._start)

        return timedelta(seconds=self._end - self._start)
