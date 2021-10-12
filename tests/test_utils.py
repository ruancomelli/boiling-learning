from unittest.case import TestCase

from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.geometry import Cylinder, Prism, RectangularPrism
from boiling_learning.utils.iterutils import (
    EvenlySpacedGoal,
    evenly_spaced_indices,
    evenly_spaced_indices_mask,
)
from boiling_learning.utils.lazy import Lazy, LazyCallable
from boiling_learning.utils.Parameters import Parameters
from boiling_learning.utils.utils import indexify


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
    def test_Prism(self):
        # consider a triangular prism

        side = 3
        length = 10

        prism = Prism(
            cross_section_perimeter=3 * side,
            cross_section_area=side ** 2 * 3 ** 0.5 / 4,
            length=length,
        )

        self.assertAlmostEqual(prism.lateral_area, 90, delta=0.1)

    def test_Cylinder(self):
        cylinder = Cylinder(length=10, diameter=2)

        self.assertEqual(cylinder.radius, 1)

    def test_RectangularPrism(self):
        prism = RectangularPrism(width=5, thickness=3, length=10)

        self.assertEqual(prism.cross_section_area, 15)
        self.assertEqual(prism.cross_section_perimeter, 16)


class Parameters_test(TestCase):
    def test_flat(self):
        p = Parameters()

        self.assertNotIn('a', p)
        with self.assertRaises(KeyError):
            p['a']

        p['a'] = 0
        self.assertIn('a', p)
        self.assertEqual(p['a'], 0)


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
        self.assertListEqual(
            evenly_spaced_indices(10, 0, goal=EvenlySpacedGoal.SPREAD), []
        )
        self.assertListEqual(
            evenly_spaced_indices(10, 1, goal=EvenlySpacedGoal.SPREAD), [5]
        )
        self.assertListEqual(
            evenly_spaced_indices(10, 2, goal=EvenlySpacedGoal.SPREAD), [3, 7]
        )
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
                    self.assertListEqual(
                        evenly_spaced_indices(total, 0, goal=goal), []
                    )

                    with self.assertRaises(ValueError):
                        evenly_spaced_indices(total, total + 1, goal=goal)

    def test_evenly_spaced_indices_mask(self) -> None:
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 0, goal=EvenlySpacedGoal.DISTANCE),
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 1, goal=EvenlySpacedGoal.DISTANCE),
            [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        )
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 2, goal=EvenlySpacedGoal.DISTANCE),
            [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 3, goal=EvenlySpacedGoal.DISTANCE),
            [
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
            ],
        )
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 5, goal=EvenlySpacedGoal.DISTANCE),
            [True, False, True, False, True, False, False, True, False, True],
        )
        self.assertListEqual(
            evenly_spaced_indices_mask(10, 10, goal=EvenlySpacedGoal.DISTANCE),
            [True, True, True, True, True, True, True, True, True, True],
        )
