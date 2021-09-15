from pathlib import Path
from unittest.case import TestCase

from tinydb import TinyDB

from boiling_learning.io.io import load_json, save_json
from boiling_learning.management.allocators import TableAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.management.Manager import Manager
from boiling_learning.management.persister import (
    FilePersister,
    FileProvider,
    Persister,
    Provider,
)
from boiling_learning.preprocessing.transformers import Creator
from boiling_learning.utils.functional import P
from boiling_learning.utils.utils import tempdir, tempfilepath


class ManagerTest(TestCase):
    def test_manager(self):
        def _load_json(path: Path):
            try:
                return True, load_json(path)
            except FileNotFoundError:
                return False, None

        with tempdir() as path:
            manager = Manager(
                path, load_method=_load_json, save_method=save_json, verbose=2
            )

            self.assertEqual(manager.table_path, path / 'lookup_table.json')

            def add(value, added):
                return value + added

            contents = P(value=5)

            with self.assertRaises(ValueError):
                manager.provide_entry(
                    contents=contents, include=False, missing_ok=False
                )

            with self.assertRaises(ValueError):
                manager.provide_elem('wololooo')

            elem_id = manager.provide_entry(
                contents=contents, include=True, missing_ok=True
            )

            res = manager.provide_elem(
                elem_id=elem_id,
                creator_params=P(value=5),
                creator=Creator(
                    'add', add, P(added=3), expand_pack_on_call=True
                ),
                save=True,
                load=True,
            )

            self.assertEqual(res, 8)
            self.assertEqual(len(manager), 1)
            self.assertTupleEqual(tuple(manager), (elem_id,))
            self.assertEqual(manager.shared_dir, manager.path / 'shared')
            self.assertEqual(
                manager.elem_workspace(elem_id),
                manager.entry_dir(elem_id) / 'workspace',
            )
            self.assertDictEqual(
                dict(manager.retrieve_elems()), {elem_id: res}
            )


class PersisterTest(TestCase):
    def test_Persister(self):
        VALUE = 'hello'
        persister = Persister(save_json, load_json)

        with tempfilepath() as filepath:
            persister.save(VALUE, filepath)
            self.assertEqual(persister.load(filepath), VALUE)

    def test_FileManager(self):
        VALUE = 3
        persister = Persister(save_json, load_json)

        with tempfilepath() as filepath:
            file_manager = FilePersister(filepath, persister)

            file_manager.save(VALUE)
            self.assertEqual(file_manager.load(), VALUE)

    def test_Provider(self):
        MISSING = 0
        provider = Provider(
            saver=save_json,
            loader=load_json,
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
            saver=save_json,
            loader=load_json,
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
    def test_TableAllocator(self) -> None:
        with tempdir() as directory:
            db = TinyDB(directory / 'db.json')
            allocator = TableAllocator(directory / 'allocator', db)

            p1 = P(3.14, 0, name='pi')
            p2 = P('hello')

            self.assertEqual(allocator(p1), allocator(p1))
            self.assertNotEqual(allocator(p1), allocator(p2))


class CacherTest(TestCase):
    def test_cache(self) -> None:
        with tempdir() as directory:
            db = TinyDB(directory / 'db.json')
            allocator = TableAllocator(directory / 'allocator', db)

            history = []

            def side_effect(number: float, name: str) -> None:
                history.append((number, name))

            @cache(allocator=allocator, saver=save_json, loader=load_json)
            def func(number: float, name: str) -> dict:
                side_effect(number, name)
                return {'number': number, 'name': name}

            self.assertListEqual(history, [])

            self.assertDictEqual(
                func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'}
            )
            self.assertListEqual(history, [(3.14, 'pi')])

            self.assertDictEqual(
                func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'}
            )
            self.assertListEqual(history, [(3.14, 'pi')])

            self.assertDictEqual(
                func(0.0, 'zero'), {'number': 0.0, 'name': 'zero'}
            )
            self.assertListEqual(history, [(3.14, 'pi'), (0.0, 'zero')])

            self.assertDictEqual(
                func(3.14, 'pi'), {'number': 3.14, 'name': 'pi'}
            )
            self.assertDictEqual(
                func(0.0, 'zero'), {'number': 0.0, 'name': 'zero'}
            )
            self.assertListEqual(history, [(3.14, 'pi'), (0.0, 'zero')])
