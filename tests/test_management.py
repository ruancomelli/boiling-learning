from pathlib import Path
from unittest.case import TestCase

from boiling_learning.io.io import load_json, save_json
from boiling_learning.management.Manager import Manager
from boiling_learning.preprocessing.transformers import Creator
from boiling_learning.utils.functional import P
from boiling_learning.utils.utils import tempdir


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
