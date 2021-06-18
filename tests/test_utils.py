from unittest.case import TestCase

from boiling_learning.utils.collections import KeyedSet
from boiling_learning.utils.geometry import Cylinder, Prism, RectangularPrism
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
