import operator
from typing import Any, Callable, Generic, Iterator, Mapping, Optional, Tuple, TypeVar, Union

import funcy
from typing_extensions import ParamSpec, Protocol

from boiling_learning.utils.dtypes import auto_dtype, new_py_function
from boiling_learning.utils.functional import Pack, nth_arg
from boiling_learning.utils.utils import JSONDataType, KeyedDefaultDict, SimpleStr

_X = TypeVar('_X')
_X_co = TypeVar('_X_co', covariant=True)
_Y_co = TypeVar('_Y_co', covariant=True)
_X1 = TypeVar('_X1')
_X2 = TypeVar('_X2')
_Y = TypeVar('_Y')
_Y1 = TypeVar('_Y1')
_Y2 = TypeVar('_Y2')
C = TypeVar('C')
T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
V = TypeVar('V')
_P = ParamSpec('_P')


class CallableWithFirst(Protocol[_X_co, _P, _Y_co]):
    def __call__(self, x: _X, *args: _P.args, **kwargs: _P.kwargs) -> _Y:
        ...


class Transformer(SimpleStr, Generic[_X, _Y]):
    def __init__(
        self, name: str, f: CallableWithFirst[_X, _P, _Y], pack: Pack[Any, Any] = Pack()
    ) -> None:
        self.__name__: str = name
        self.pack: Pack[Any, Any] = pack
        self._call = pack.rpartial(f)

    @property
    def name(self) -> str:
        return self.__name__

    def __call__(self, arg: _X, *args: Any, **kwargs: Any) -> _Y:
        return self._call(arg, *args, **kwargs)

    @classmethod
    def make(cls: Callable[..., C], name: str, *args, **kwargs) -> Callable[[Callable], C]:
        def _make(f: Callable) -> C:
            return cls(name, f, *args, **kwargs)

        return _make

    def __describe__(self) -> JSONDataType:
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'pack': self.pack,
        }

    def as_tf_py_function(self, pack_tuple: bool = False):
        if pack_tuple:

            def func(*a):
                return self(a)

        else:
            func = self

        def _tf_py_function(*args):
            return new_py_function(func=func, inp=args, Tout=auto_dtype(args))

        return _tf_py_function


class Creator(Transformer[Pack, _Y], Generic[_Y]):
    def __init__(
        self,
        name: str,
        f: Callable[..., _Y],
        pack: Pack = Pack(),
        expand_pack_on_call: bool = False,
    ) -> None:
        if expand_pack_on_call:

            def g(pack: Pack, *args, **kwargs) -> _Y:
                return f(*(pack.args + args), **{**pack.kwargs, **kwargs})

        else:
            g = f

        super().__init__(name, g, pack=pack)


class FeatureTransformer(Transformer[Tuple[_X1, _Y], Tuple[_X2, _Y]], Generic[_X1, _X2, _Y]):
    def __init__(self, name: str, f: Callable[..., _X2], pack: Pack = Pack()) -> None:
        def g(pair: Tuple[_X1, _Y], *args, **kwargs) -> Tuple[_X2, _Y]:
            def pair_transformer(feature: _X1, target: _Y) -> Tuple[_X2, _Y]:
                return f(feature, *args, **kwargs), target

            return pair_transformer(*pair)

        super().__init__(name, g, pack=pack)

    def transform_feature(self, feature: _X1, *args, **kwargs) -> _X2:
        return self((feature, None), *args, **kwargs)[0]


class PairTransformer(Transformer[Tuple[_X1, _Y1], Tuple[_X2, _Y2]], Generic[_X1, _Y1, _X2, _Y2]):
    def __init__(
        self,
        name: str,
        feature_transformer: Transformer[_X1, _X2],
        target_transformer: Transformer[_Y1, _Y2],
    ) -> None:
        def f(feature: _X1, target: _Y1) -> Tuple[_X2, _Y2]:
            return feature_transformer(feature), target_transformer(target)

        super().__init__(name, f)


class KeyedFeatureTransformer(
    Transformer[Tuple[_X1, _Y], Tuple[Union[_X1, _X2], _Y]],
    Generic[_X1, _X2, _Y],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., _X2],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
        key_getter: Callable[[_Y], Optional[str]] = operator.itemgetter('name'),
    ) -> None:
        self.packer = packer

        def g(pair: Tuple[_X1, _Y], *args, **kwargs) -> Tuple[Union[_X1, _X2], _Y]:
            def mapped_f(feature: _X1, target: _Y) -> Tuple[Union[_X1, _X2], _Y]:
                key = key_getter(target)
                featre_transformer = self.get_feature_transformer(f, key)
                return featre_transformer(feature), target

            return mapped_f(*pair)

        super().__init__(name, g)

    def get_feature_transformer(
        self, f: Callable[..., _X2], key: Optional[str]
    ) -> Callable[[_X1], Union[_X1, _X2]]:
        if callable(self.packer):
            pack = self.packer(key)
            return pack.rpartial(f)

        if key in self.packer:
            return self._get_partial_transformer(f, key)

        if None in self.packer:
            return self._get_partial_transformer(f, None)

        return funcy.identity

    def _get_partial_transformer(
        self, f: Callable[..., _X2], key: Optional[str]
    ) -> Callable[[_X1], _X2]:
        return self.packer[key].rpartial(f)

    def __describe__(self) -> JSONDataType:
        return funcy.merge(super().__describe__(), {'packer': self.packer})


class DictFeatureTransformer(
    Mapping[str, Transformer[Tuple[_X1, _Y], Tuple[Union[_X1, _X2], _Y]]],
    Generic[_X1, _X2, _Y],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., _X2],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
    ) -> None:
        self.__name__: str = name
        self.packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]] = packer
        self._transformer_mapping: KeyedDefaultDict[
            str, FeatureTransformer[_X1, _X2, _Y]
        ] = KeyedDefaultDict(self._transformer_factory)
        self.func: Callable[..., _X2] = f

    @property
    def name(self) -> str:
        return self.__name__

    def _resolve_func_and_pack(self, key: str) -> Tuple[Callable[[_X1], _X2], Pack]:
        if isinstance(self.packer, Mapping):
            try:
                if key in self.packer:
                    return self.func, self.packer[key]

                pack = self.packer[None]

                if pack is None:
                    return funcy.identity, Pack()
                else:
                    return self.func, pack
            except KeyError as e:
                raise KeyError(
                    f'Invalid key {key}: corresponding pack was not found.'
                    ' Define a default pack by passing None: default_pack'
                    ' or None: None to skip missing keys.'
                ) from e

        elif callable(self.packer):
            return self.func, self.packer(key)
        else:
            raise TypeError(
                'self.packer must be either a Mapping[Optional[str], Pack] '
                'or a Callable[[str], Pack]. '
                f'Got {type(self.packer)}'
            )

    def _transformer_factory(self, key: str) -> FeatureTransformer[_X1, _X2, _Y]:
        name = '_'.join((self.name, key))
        func, pack = self._resolve_func_and_pack(key)
        return FeatureTransformer(name, func, pack)

    def __iter__(self) -> Iterator[str]:
        return iter(self._transformer_mapping)

    def __len__(self) -> int:
        return len(self._transformer_mapping)

    def __getitem__(self, key: str) -> FeatureTransformer[_X1, _X2, _Y]:
        return self._transformer_mapping[key]

    def __describe__(self) -> JSONDataType:
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'packer': self.packer if isinstance(self.packer, Mapping) else self.packer.__name__,
        }


first_argument_transformer = Transformer('first_argument', nth_arg(0))


def transformer(*args, **kwargs) -> Callable[[Callable], Transformer]:
    def decorator(f: Callable) -> Transformer:
        maker = Transformer.make(f.__name__, *args, **kwargs)
        return maker(f)

    return decorator


def creator(*args, **kwargs) -> Callable[[Callable[..., S]], Creator[S]]:
    def decorator(f: Callable[..., S]) -> Creator[S]:
        maker = Creator.make(f.__name__, *args, **kwargs)
        return maker(f)

    return decorator
