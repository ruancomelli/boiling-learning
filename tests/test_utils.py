from unittest.case import TestCase

from boiling_learning.utils.collections import KeyedSet
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
