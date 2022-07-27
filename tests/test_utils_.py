import math

from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.geometry import Cylinder, RectangularPrism
from boiling_learning.utils.iterutils import unsort
from boiling_learning.utils.lazy import Lazy, LazyCallable


class Test_utils_collections:
    def test_KeyedSet(self) -> None:
        keyed_set = KeyedSet(str.upper, ('hi', 'bye', 'hello'))

        assert set(keyed_set) == {'hi', 'bye', 'hello'}
        assert set(keyed_set.values()) == {'hi', 'bye', 'hello'}
        assert set(keyed_set.keys()) == {'HI', 'BYE', 'HELLO'}
        assert len(keyed_set) == 3
        assert keyed_set['BYE'] == 'bye'
        assert 'hello' in keyed_set
        assert 'byello' not in keyed_set

        keyed_set.add('byello')
        assert 'byello' in keyed_set

        keyed_set.discard('hello')
        assert 'hello' not in keyed_set


class Test_geometry:
    def test_Cylinder(self) -> None:
        cylinder = Cylinder(length=10, diameter=2)

        assert cylinder.radius() == 1
        assert cylinder.volume() == math.pi * 1 ** 2 * 10

    def test_RectangularPrism(self) -> None:
        prism = RectangularPrism(width=5, thickness=3, length=10)

        assert prism.cross_section_area() == 15
        assert prism.cross_section_perimeter() == 16


class TestLazy:
    def test_Lazy(self) -> None:
        history = []

        def creator() -> int:
            history.append(0)  # simulate a side-effect
            return 0

        assert history == []
        lazy_number = Lazy(creator)
        assert history == []
        assert lazy_number() == 0
        assert history == [0]

        lazy_number = Lazy.from_value(1)
        assert lazy_number() == 1

    def test_LazyCallable(self) -> None:
        history = []

        def add(left: int, right: int) -> int:
            addition = left + right
            history.append(addition)  # simulate a side-effect
            return addition

        lazy_add = LazyCallable(add)

        assert history == []

        result = lazy_add(1, 2)
        assert history == []

        assert result() == 3
        assert history == [3]


class Test_iterutils:
    def test_unsort(self) -> None:
        unsorted_items = [50, 30, 20, 10, 40]

        unsorters, sorted_items = unsort(unsorted_items)
        unsorters = list(unsorters)
        sorted_items = list(sorted_items)
        assert sorted_items == [10, 20, 30, 40, 50]
        assert unsorters == [4, 2, 1, 0, 3]

        assert [sorted_items[index] for index in unsorters] == unsorted_items

        assert unsort(()) == ((), ())
