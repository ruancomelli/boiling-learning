import math
from pathlib import Path
from typing import List

import pytest

from boiling_learning.descriptions import describe
from boiling_learning.io import json
from boiling_learning.management.allocators import JSONTableAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.management.persister import Persister, provide
from boiling_learning.utils.functional import P


@pytest.fixture
def filepath(tmp_path: Path) -> Path:
    return tmp_path / 'file'


class TestPersister:
    def test_Persister(self, filepath: Path):
        VALUE = 'hello'
        persister = Persister(json.dump, json.load)

        persister.save(VALUE, filepath)
        assert persister.load(filepath) == VALUE

    def test_provide(self, filepath: Path):
        MISSING = 0

        def creator() -> int:
            return MISSING

        persister = Persister(json.dump, json.load)

        VALUE = 3
        persister.save(VALUE, filepath)
        assert filepath.is_file()
        assert (
            provide(
                filepath,
                creator=creator,
                persister=persister,
            )
            == VALUE
        )

        filepath.unlink()

        assert not filepath.is_file()
        assert (
            provide(
                filepath,
                creator=creator,
                persister=persister,
            )
            == MISSING
        )


class TestAllocators:
    def test_JSONAllocator(self, tmp_path: Path) -> None:
        allocator = JSONTableAllocator(tmp_path)

        p1 = P('3.14', 0, name='pi')
        p2 = P('hello')

        assert allocator(p1) == allocator(p1)
        assert allocator(P('3.14', 0, name='pi')) == allocator(p1)
        assert allocator(p1) != allocator(p2)
        assert allocator.allocate('3.14', 0, name='pi') == allocator(p1)
        assert allocator.allocate('hello') == allocator(p2)


class TestCacher:
    def test_cache(self, tmp_path: Path) -> None:
        allocator = JSONTableAllocator(tmp_path)

        history = []

        def side_effect(number: float, name: str) -> None:
            history.append((number, name))

        @cache(allocator=allocator, saver=json.dump, loader=json.load)
        def func(number: float, name: str) -> dict:
            side_effect(number, name)
            return {'number': number, 'name': name}

        assert not history

        assert func(3.14, 'pi') == {'number': 3.14, 'name': 'pi'}
        assert history == [(3.14, 'pi')]

        assert func(3.14, 'pi') == {'number': 3.14, 'name': 'pi'}
        assert history == [(3.14, 'pi')]

        assert func(0.0, 'zero') == {'number': 0.0, 'name': 'zero'}
        assert history == [(3.14, 'pi'), (0.0, 'zero')]

        assert func(3.14, 'pi') == {'number': 3.14, 'name': 'pi'}
        assert func(0.0, 'zero') == {'number': 0.0, 'name': 'zero'}
        assert history == [(3.14, 'pi'), (0.0, 'zero')]


class TestDescriptor:
    def test_describe(self) -> None:
        assert describe.supports(None)
        assert describe(None) is None

        assert describe.supports(5)
        assert describe(5) == 5

        assert describe.supports(3.14)
        assert math.isclose(describe(3.14), 3.14, abs_tol=1e-8)

        assert describe.supports('hello')
        assert describe('hello') == 'hello'

        class CrazyItems:
            def __init__(self, length: int) -> None:
                self.value: int = length

            def __describe__(self) -> List[int]:
                return [self.value] * self.value

        assert describe.supports(CrazyItems(3))
        assert describe(CrazyItems(3)) == [3, 3, 3]
