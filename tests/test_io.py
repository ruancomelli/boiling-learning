from typing import Any, Dict
from unittest import TestCase

import pytest

from boiling_learning.io import json
from boiling_learning.io.json import JSONDataType


class X:
    def __init__(self, value: int) -> None:
        self.value: int = value

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, X) and self.value == other.value


@json.encode.instance(X)
def _json_encode(obj: X) -> Dict[str, int]:
    return {'value': obj.value}


@json.decode.dispatch(X)
def _json_decode(obj: JSONDataType) -> X:
    return X(obj['value'])


class storage_Test(TestCase):
    def test_default_json_io(self):
        test_list = [
            314159,
            'apple pie tastes good',
            {'likes dinos': True, 'political opinion': None},
        ]

        encoded = json.dumps(json.serialize(test_list))
        decoded = json.deserialize(json.loads(encoded))

        self.assertListEqual(decoded, test_list)

    def test_custom_json_io(self):
        x = X(3)

        encoded = json.dumps(json.serialize(x))
        decoded = json.deserialize(json.loads(encoded))

        self.assertEqual(decoded.value, 3)


@pytest.mark.parametrize(
    'obj,encoded',
    [
        pytest.param(None, None),
        pytest.param(3, 3),
        pytest.param(3.14, 3.14),
        pytest.param('hello', 'hello'),
        pytest.param(True, True),
        pytest.param([], [], id='empty-list'),
        pytest.param(X(5), {'value': 5}, id='custom-type'),
        pytest.param(
            [0, 1, 'hello'],
            [
                {'type': 'builtins.int', 'contents': 0},
                {'type': 'builtins.int', 'contents': 1},
                {'type': 'builtins.str', 'contents': 'hello'},
            ],
            id='list-of-basics',
        ),
        pytest.param(
            (0, None, 'hello'),
            [
                {'type': 'builtins.int', 'contents': 0},
                {'type': None, 'contents': None},
                {'type': 'builtins.str', 'contents': 'hello'},
            ],
            id='tuple-of-basics',
        ),
        pytest.param(
            (3, 'hi', ['no', 'yes'], True, None),
            [
                {'type': 'builtins.int', 'contents': 3},
                {'type': 'builtins.str', 'contents': 'hi'},
                {
                    'type': 'builtins.list',
                    'contents': [
                        {'type': 'builtins.str', 'contents': 'no'},
                        {'type': 'builtins.str', 'contents': 'yes'},
                    ],
                },
                {'type': 'builtins.bool', 'contents': True},
                {'type': None, 'contents': None},
            ],
            id='tuple-of-basics',
        ),
        pytest.param(
            (3, 'hi', ('no', X(4)), True, None),
            [
                {'type': 'builtins.int', 'contents': 3},
                {'type': 'builtins.str', 'contents': 'hi'},
                {
                    'type': 'builtins.tuple',
                    'contents': [
                        {'type': 'builtins.str', 'contents': 'no'},
                        {'type': f'{__name__}.X', 'contents': {'value': 4}},
                    ],
                },
                {'type': 'builtins.bool', 'contents': True},
                {'type': None, 'contents': None},
            ],
            id='tuple-of-complex',
        ),
    ],
)
def test_json_encode_decode(obj: Any, encoded: Any) -> None:
    encoded_obj = json.encode(obj)
    assert encoded_obj == encoded
    assert json.decode[type(obj) if obj is not None else None](encoded_obj) == obj


@pytest.mark.parametrize(
    'obj,serialized',
    [
        pytest.param(None, {'type': None, 'contents': None}),
        pytest.param(3, {'type': 'builtins.int', 'contents': 3}),
        pytest.param(3.14, {'type': 'builtins.float', 'contents': 3.14}),
        pytest.param('hello', {'type': 'builtins.str', 'contents': 'hello'}),
        pytest.param(True, {'type': 'builtins.bool', 'contents': True}),
        pytest.param([], {'type': 'builtins.list', 'contents': []}, id='empty-list'),
        pytest.param(X(5), {'type': f'{__name__}.X', 'contents': {'value': 5}}, id='custom-type'),
        pytest.param(
            [0, 1, 'hello'],
            {
                'type': 'builtins.list',
                'contents': [
                    {'type': 'builtins.int', 'contents': 0},
                    {'type': 'builtins.int', 'contents': 1},
                    {'type': 'builtins.str', 'contents': 'hello'},
                ],
            },
            id='list-of-basics',
        ),
        pytest.param(
            (0, None, 'hello'),
            {
                'type': 'builtins.tuple',
                'contents': [
                    {'type': 'builtins.int', 'contents': 0},
                    {'type': None, 'contents': None},
                    {'type': 'builtins.str', 'contents': 'hello'},
                ],
            },
            id='tuple-of-basics',
        ),
        pytest.param(
            (3, 'hi', ['no', 'yes'], True, None),
            {
                'type': 'builtins.tuple',
                'contents': [
                    {'type': 'builtins.int', 'contents': 3},
                    {'type': 'builtins.str', 'contents': 'hi'},
                    {
                        'type': 'builtins.list',
                        'contents': [
                            {'type': 'builtins.str', 'contents': 'no'},
                            {'type': 'builtins.str', 'contents': 'yes'},
                        ],
                    },
                    {'type': 'builtins.bool', 'contents': True},
                    {'type': None, 'contents': None},
                ],
            },
            id='tuple-of-complex-native',
        ),
        pytest.param(
            (3, 'hi', ('no', X(4)), True, None),
            {
                'type': 'builtins.tuple',
                'contents': [
                    {'type': 'builtins.int', 'contents': 3},
                    {'type': 'builtins.str', 'contents': 'hi'},
                    {
                        'type': 'builtins.tuple',
                        'contents': [
                            {'type': 'builtins.str', 'contents': 'no'},
                            {'type': f'{__name__}.X', 'contents': {'value': 4}},
                        ],
                    },
                    {'type': 'builtins.bool', 'contents': True},
                    {'type': None, 'contents': None},
                ],
            },
            id='tuple-of-complex-custom',
        ),
    ],
)
def test_serialization_deserialization(obj: Any, serialized: Any) -> None:
    serialized_obj = json.serialize(obj)
    assert serialized_obj == serialized
    assert json.deserialize(serialized_obj) == obj
