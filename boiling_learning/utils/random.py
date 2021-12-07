import random
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def random_state(seed_value: Optional[int] = None) -> Iterator[None]:
    state = random.getstate()
    random.seed(seed_value)

    try:
        yield
    finally:
        random.setstate(state)
