from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, Mapping, Optional, TypeVar

from typing_extensions import Protocol

from boiling_learning.io import json
from boiling_learning.utils import KeyedDefaultDict, SimpleStr
from boiling_learning.utils.descriptions import describe
from boiling_learning.utils.functional import Pack

_X_contra = TypeVar('_X_contra', contravariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_X = TypeVar('_X')
_X1 = TypeVar('_X1')
_X2 = TypeVar('_X2')
_Y = TypeVar('_Y')


class Transformer(SimpleStr, Generic[_X, _Y]):
    def __init__(
        self,
        name: str,
        f: CallableWithFirst[_X, _Y],
        pack: Pack[Any, Any] = Pack(),
    ) -> None:
        self.__name__ = name
        self._call: Callable[[_X], _Y] = pack.rpartial(f)
        self.pack: Pack[Any, Any] = pack

    @property
    def name(self) -> str:
        return self.__name__

    def __call__(self, arg: _X) -> _Y:
        return self._call(arg)

    def __describe__(self) -> json.JSONDataType:
        return json.serialize(
            {
                'type': self.__class__.__name__,
                'name': self.name,
                'pack': self.pack,
            }
        )


class DictTransformer(
    Mapping[str, Transformer[_X1, _X2]],
    Generic[_X1, _X2],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., _X2],
        packer: Mapping[Optional[str], Pack],
    ) -> None:
        self.__name__: str = name
        self.packer: Mapping[Optional[str], Pack] = packer
        self._transformer_mapping: KeyedDefaultDict[str, Transformer[_X1, _X2]] = KeyedDefaultDict(
            lambda key: Transformer(f'{self.name}_{key}', self.func, self.packer[key])
        )
        self.func: Callable[..., _X2] = f

    @property
    def name(self) -> str:
        return self.__name__

    def __iter__(self) -> Iterator[str]:
        return iter(self._transformer_mapping)

    def __len__(self) -> int:
        return len(self._transformer_mapping)

    def __getitem__(self, key: str) -> Transformer[_X1, _X2]:
        return self._transformer_mapping[key]

    def __describe__(self) -> json.JSONDataType:
        return json.serialize(
            {
                'type': self.__class__.__name__,
                'name': self.name,
                'packer': self.packer,
            }
        )


@json.encode.instance(Transformer)
def _encode_transformer(instance: Transformer[Any, Any]) -> json.JSONDataType:
    return json.serialize(describe(instance))


@json.encode.instance(DictTransformer)
def _encode_dict_feature_transformer(instance: DictTransformer[Any, Any]) -> json.JSONDataType:
    return json.serialize(describe(instance))


class CallableWithFirst(Protocol[_X_contra, _Y_co]):
    def __call__(self, first: _X_contra, *args: Any, **kwargs: Any) -> _Y_co:
        ...
