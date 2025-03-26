import random
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def random_state(seed_value: int | None = None) -> Iterator[None]:
    state = random.getstate()
    random.seed(seed_value)

    try:
        yield
    finally:
        random.setstate(state)
