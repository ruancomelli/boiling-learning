from contextlib import contextmanager
from datetime import timedelta
from timeit import default_timer
from types import TracebackType
from typing import Dict, Iterator, Optional, Type


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
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._end = default_timer()

    @property
    def duration(self) -> Optional[timedelta]:
        if self._start is None:
            return None

        if self._end is None:
            return timedelta(seconds=default_timer() - self._start)

        return timedelta(seconds=self._end - self._start)


class CasesTimer(Dict[str, timedelta]):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name

    @contextmanager
    def case(self, name: str) -> Iterator[Timer]:
        with Timer() as t:
            yield t

        duration = t.duration

        if duration is not None:
            self[name] = timedelta(seconds=duration)

    def pretty(self) -> str:
        return '\n\t'.join(
            (
                f'{self.name}:',
                *(f'{case_name}: {timing}' for case_name, timing in self.items()),
                '',
                f'Total: {sum(self.values(), start=timedelta())}',
            )
        )
