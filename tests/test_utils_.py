import math
from unittest.case import TestCase

from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.geometry import Cylinder, RectangularPrism
from boiling_learning.utils.iterutils import evenly_spaced_indices
from boiling_learning.utils.lazy import Lazy, LazyCallable
from boiling_learning.utils.utils import indexify, unsort


class utils_utils_test(TestCase):
    def test_indexify(self) -> None:
        self.assertEqual(tuple(indexify('abc')), (0, 1, 2))

    def test_unsort(self) -> None:
        unsorted_items = [50, 30, 20, 10, 40]

        unsorters, sorted_items = unsort(unsorted_items)
        unsorters = list(unsorters)
        sorted_items = list(sorted_items)
        assert sorted_items == [10, 20, 30, 40, 50]
        assert unsorters == [4, 2, 1, 0, 3]

        assert [sorted_items[index] for index in unsorters] == unsorted_items

        assert unsort(()) == ((), ())


class utils_collections_test(TestCase):
    def test_KeyedSet(self) -> None:
        keyed_set = KeyedSet(str.upper, ('hi', 'bye', 'hello'))

        self.assertEqual(set(keyed_set), {'hi', 'bye', 'hello'})
        self.assertEqual(set(keyed_set.values()), {'hi', 'bye', 'hello'})
        self.assertEqual(set(keyed_set.keys()), {'HI', 'BYE', 'HELLO'})
        self.assertEqual(len(keyed_set), 3)
        self.assertEqual(keyed_set['BYE'], 'bye')
        self.assertIn('hello', keyed_set)
        self.assertNotIn('byello', keyed_set)

        keyed_set.add('byello')
        self.assertIn('byello', keyed_set)

        keyed_set.discard('hello')
        self.assertNotIn('hello', keyed_set)


class geometry_test(TestCase):
    def test_Cylinder(self) -> int:
        cylinder = Cylinder(length=10, diameter=2)

        assert cylinder.radius() == 1
        assert cylinder.volume() == math.pi * 1 ** 2 * 10

    def test_RectangularPrism(self) -> int:
        prism = RectangularPrism(width=5, thickness=3, length=10)

        self.assertEqual(prism.cross_section_area(), 15)
        self.assertEqual(prism.cross_section_perimeter(), 16)


class LazyTest(TestCase):
    def test_Lazy(self) -> None:
        history = []

        def creator() -> int:
            history.append(0)  # simulate a side-effect
            return 0

        self.assertListEqual(history, [])
        lazy_number = Lazy(creator)
        self.assertListEqual(history, [])
        self.assertEqual(lazy_number(), 0)
        self.assertListEqual(history, [0])

        lazy_number = Lazy.from_value(1)
        self.assertEqual(lazy_number(), 1)

    def test_LazyCallable(self) -> None:
        history = []

        def add(left: int, right: int) -> int:
            addition = left + right
            history.append(addition)  # simulate a side-effect
            return addition

        lazy_add = LazyCallable(add)

        self.assertListEqual(history, [])

        result = lazy_add(1, 2)
        self.assertListEqual(history, [])

        self.assertEqual(result(), 3)
        self.assertListEqual(history, [3])


class iterutils_test(TestCase):
    def test_evenly_spaced_indices(self) -> None:
        self.assertListEqual(evenly_spaced_indices(10, 0, goal='spread'), [])
        self.assertListEqual(evenly_spaced_indices(10, 1, goal='spread'), [5])
        self.assertListEqual(evenly_spaced_indices(10, 2, goal='spread'), [3, 7])
        self.assertListEqual(evenly_spaced_indices(10, 3, goal='spread'), [2, 5, 8])
        self.assertListEqual(evenly_spaced_indices(10, 5, goal='spread'), [2, 3, 5, 7, 8])
        self.assertListEqual(
            evenly_spaced_indices(10, 10, goal='spread'), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        for total, count in ((10, 6), (100, 70), (100, 5)):
            with self.subTest('General properties', total=total, count=count):
                self.assertListEqual(evenly_spaced_indices(total, 0, goal='distance'), [])

                with self.assertRaises(ValueError):
                    evenly_spaced_indices(total, total + 1, goal='distance')

                self.assertListEqual(evenly_spaced_indices(total, 0, goal='spread'), [])

                with self.assertRaises(ValueError):
                    evenly_spaced_indices(total, total + 1, goal='spread')
