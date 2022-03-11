from typing import List
from unittest.case import TestCase

from tinydb import TinyDB

from boiling_learning.io import json
from boiling_learning.management.allocators import JSONTableAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.management.persister import FilePersister, FileProvider, Persister, Provider
from boiling_learning.utils import tempdir, tempfilepath
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.functional import P


class PersisterTest(TestCase):
    def test_Persister(self):
        VALUE = 'hello'
        persister = Persister(json.dump, json.load)

        with tempfilepath() as filepath:
            persister.save(VALUE, filepath)
            self.assertEqual(persister.load(filepath), VALUE)

    def test_FileManager(self):
        VALUE = 3
        persister = Persister(json.dump, json.load)

        with tempfilepath() as filepath:
            file_manager = FilePersister(filepath, persister)

            file_manager.save(VALUE)
            self.assertEqual(file_manager.load(), VALUE)

    def test_Provider(self):
        MISSING = 0
        provider = Provider(
            saver=json.dump,
            loader=json.load,
            creator=lambda: MISSING,
        )

        with tempfilepath() as filepath:
            VALUE = 3
            provider.save(VALUE, filepath)
            self.assertTrue(filepath.is_file())
            self.assertEqual(provider.provide(filepath), VALUE)

        self.assertFalse(filepath.is_file())
        self.assertEqual(provider.provide(filepath), MISSING)

    def test_FileProvider(self):
        MISSING = 0
        provider = Provider(
            saver=json.dump,
            loader=json.load,
            creator=lambda: MISSING,
        )

        with tempfilepath() as filepath:
            provider = FileProvider(filepath, provider)

            VALUE = 3
            provider.save(VALUE)
            self.assertTrue(filepath.is_file())
            self.assertEqual(provider.provide(), VALUE)

        self.assertFalse(filepath.is_file())
        self.assertEqual(provider.provide(), MISSING)


class AllocatorsTest(TestCase):
    def test_JSONAllocator(self) -> None:
        with tempdir() as directory:
            db = TinyDB(directory / 'db.json')
            allocator = JSONTableAllocator(directory / 'allocator', db)

            p1 = P(3.14, 0, name='pi')
            p2 = P('hello')

            self.assertEqual(allocator(p1), allocator(p1))
            self.assertNotEqual(allocator(p1), allocator(p2))


class CacherTest(TestCase):
    def test_cache(self) -> None:
        with tempdir() as directory:
            db = TinyDB(directory / 'db.json')
            allocator = JSONTableAllocator(directory / 'allocator', db)

            history = []

            def side_effect(number: float, name: str) -> None:
                history.append((number, name))

            @cache(allocator=allocator, saver=json.dump, loader=json.load)
            def func(number: float, name: str) -> dict:
                side_effect(number, name)
                return {'number': number, 'name': name}

            self.assertListEqual(history, [])

            self.assertDictEqual(func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'})
            self.assertListEqual(history, [(3.14, 'pi')])

            self.assertDictEqual(func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'})
            self.assertListEqual(history, [(3.14, 'pi')])

            self.assertDictEqual(func(0.0, 'zero'), {'number': 0.0, 'name': 'zero'})
            self.assertListEqual(history, [(3.14, 'pi'), (0.0, 'zero')])

            self.assertDictEqual(func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'})
            self.assertDictEqual(func(0.0, 'zero'), {'number': 0.0, 'name': 'zero'})
            self.assertListEqual(history, [(3.14, 'pi'), (0.0, 'zero')])


class DescriptorTest(TestCase):
    def test_describe(self) -> None:
        self.assertTrue(describe.supports(None))
        self.assertIsNone(describe(None))

        self.assertTrue(describe.supports(5))
        self.assertEqual(describe(5), 5)

        self.assertTrue(describe.supports(3.14))
        self.assertAlmostEqual(describe(3.14), 3.14, delta=1e-8)

        self.assertTrue(describe.supports('hello'))
        self.assertEqual(describe('hello'), 'hello')

        class CrazyItems:
            def __init__(self, length: int) -> None:
                self.value: int = length

            def __describe__(self) -> List[int]:
                return [self.value] * self.value

        self.assertTrue(describe.supports(CrazyItems(3)))
        self.assertListEqual(describe(CrazyItems(3)), [3, 3, 3])
