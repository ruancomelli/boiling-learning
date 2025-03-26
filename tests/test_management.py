import math
from pathlib import Path

import pytest

from boiling_learning.descriptions import describe
from boiling_learning.io.json import dump as saver
from boiling_learning.io.json import load as loader
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import Cacher, cache
from boiling_learning.utils.functional import P


@pytest.fixture
def filepath(tmp_path: Path) -> Path:
    return tmp_path / "file"


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def allocator(cache_path: Path) -> JSONAllocator:
    return JSONAllocator(cache_path)


class TestAllocators:
    def test_JSONAllocator(self, allocator: JSONAllocator) -> None:
        p1 = P("3.14", 0, name="pi")
        p2 = P("hello")

        assert allocator(p1) == allocator(p1)
        assert allocator(P("3.14", 0, name="pi")) == allocator(p1)
        assert allocator(p1) != allocator(p2)
        assert allocator.allocate("3.14", 0, name="pi") == allocator(p1)
        assert allocator.allocate("hello") == allocator(p2)


class TestCacher:
    def test_provide(self, allocator: JSONAllocator, filepath: Path):
        cacher = Cacher(allocator, saver=saver, loader=loader)

        MISSING = 0

        def creator() -> int:
            return MISSING

        VALUE = 3
        saver(VALUE, filepath)
        assert filepath.is_file()
        assert cacher.provide(creator, filepath) == VALUE

        filepath.unlink()

        assert not filepath.is_file()
        assert cacher.provide(creator, filepath) == MISSING

    def test_cache(self, allocator: JSONAllocator) -> None:
        history = []

        def side_effect(number: float, name: str) -> None:
            history.append((number, name))

        @cache(allocator=allocator, saver=saver, loader=loader)
        def func(number: float, name: str) -> dict:
            side_effect(number, name)
            return {"number": number, "name": name}

        assert not history

        assert func(3.14, "pi") == {"number": 3.14, "name": "pi"}
        assert history == [(3.14, "pi")]

        assert func(3.14, "pi") == {"number": 3.14, "name": "pi"}
        assert history == [(3.14, "pi")]

        assert func(0.0, "zero") == {"number": 0.0, "name": "zero"}
        assert history == [(3.14, "pi"), (0.0, "zero")]

        assert func(3.14, "pi") == {"number": 3.14, "name": "pi"}
        assert func(0.0, "zero") == {"number": 0.0, "name": "zero"}
        assert history == [(3.14, "pi"), (0.0, "zero")]


class TestDescriptor:
    def test_describe(self) -> None:
        assert describe.supports(None)
        assert describe(None) is None

        assert describe.supports(5)
        assert describe(5) == 5

        assert describe.supports(3.14)
        assert math.isclose(describe(3.14), 3.14, abs_tol=1e-8)

        assert describe.supports("hello")
        assert describe("hello") == "hello"

        class CrazyItems:
            def __init__(self, length: int) -> None:
                self.value: int = length

            def __describe__(self) -> list[int]:
                return [self.value] * self.value

        assert describe.supports(CrazyItems(3))
        assert describe(CrazyItems(3)) == [3, 3, 3]
