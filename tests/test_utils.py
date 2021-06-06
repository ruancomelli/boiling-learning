from boiling_learning.utils.utils import indexify
from unittest.case import TestCase


class utils_utils_test(TestCase):
    def test_indexify(self):
        self.assertEqual(
            tuple(indexify('abc')),
            (0, 1, 2)
        )
