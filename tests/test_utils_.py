import math
from unittest.case import TestCase

from boiling_learning.utils import indexify
from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.geometry import Cylinder, RectangularPrism
from boiling_learning.utils.iterutils import EvenlySpacedGoal, evenly_spaced_indices
from boiling_learning.utils.lazy import Lazy, LazyCallable


class utils_utils_test(TestCase):
    def test_indexify(self):
        self.assertEqual(tuple(indexify('abc')), (0, 1, 2))


class utils_collections_test(TestCase):
    def test_KeyedSet(self):
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
    def test_Cylinder(self):
        cylinder = Cylinder(length=10, diameter=2)

        assert cylinder.radius() == 1
        assert cylinder.volume() == math.pi * 1 ** 2 * 10

    def test_RectangularPrism(self):
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
        self.assertListEqual(evenly_spaced_indices(10, 0, goal=EvenlySpacedGoal.SPREAD), [])
        self.assertListEqual(evenly_spaced_indices(10, 1, goal=EvenlySpacedGoal.SPREAD), [5])
        self.assertListEqual(evenly_spaced_indices(10, 2, goal=EvenlySpacedGoal.SPREAD), [3, 7])
        self.assertListEqual(
            evenly_spaced_indices(10, 3, goal=EvenlySpacedGoal.SPREAD),
            [2, 5, 8],
        )
        self.assertListEqual(
            evenly_spaced_indices(10, 5, goal=EvenlySpacedGoal.SPREAD),
            [2, 3, 5, 7, 8],
        )
        self.assertListEqual(
            evenly_spaced_indices(10, 10, goal=EvenlySpacedGoal.SPREAD),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

        for total, count in ((10, 6), (100, 70), (100, 5)):
            with self.subTest('General properties', total=total, count=count):
                for goal in EvenlySpacedGoal:
                    self.assertListEqual(evenly_spaced_indices(total, 0, goal=goal), [])

                    with self.assertRaises(ValueError):
                        evenly_spaced_indices(total, total + 1, goal=goal)
