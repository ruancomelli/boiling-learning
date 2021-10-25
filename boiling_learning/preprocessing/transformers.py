import operator
from typing import (
    Callable,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import funcy

from boiling_learning.utils.dtypes import auto_dtype, new_py_function
from boiling_learning.utils.functional import Pack, nth_arg
from boiling_learning.utils.utils import (
    FrozenNamedMixin,
    JSONDataType,
    KeyedDefaultDict,
    SimpleStr,
)

_X = TypeVar('_X')
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


class Transformer(FrozenNamedMixin, SimpleStr, Generic[_X, _Y]):
    def __init__(self, name: str, f: Callable[..., _Y], pack: Pack = Pack()):
        super().__init__(name)

        self.pack = pack
        self.transformer = pack.rpartial(f)

    def __call__(self, arg: _X, *args, **kwargs) -> _Y:
        return self.transformer(arg, *args, **kwargs)

    @classmethod
    def make(
        cls: Callable[..., C], name: str, *args, **kwargs
    ) -> Callable[[Callable], C]:
        def _make(f: Callable) -> C:
            return cls(name, f, *args, **kwargs)

        return _make

    def describe(self) -> JSONDataType:
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
    ):
        if expand_pack_on_call:

            def g(pack: Pack, *args, **kwargs) -> _Y:
                return f(*(pack.args + args), **{**pack.kwargs, **kwargs})

        else:
            g = f

        super().__init__(name, g, pack=pack)


class ImageTransformer(
    Transformer[Tuple[_X1, _Y], Tuple[_X2, _Y]], Generic[_X1, _X2, _Y]
):
    def __init__(self, name: str, f: Callable[..., _X2], pack: Pack = Pack()):
        def g(
            img_data_pair: Tuple[_X1, _Y], *args, **kwargs
        ) -> Tuple[_X2, _Y]:
            def pair_transformer(img: _X1, data: _Y) -> Tuple[_X2, _Y]:
                return f(img, *args, **kwargs), data

            return pair_transformer(*img_data_pair)

        super().__init__(name, g, pack=pack)

    def transform_image(self, img: _X1, *args, **kwargs) -> _X2:
        return self((img, None), *args, **kwargs)[0]


class ImageDatasetTransformer(
    Transformer[Tuple[_X1, _Y1], Tuple[_X2, _Y2]], Generic[_X1, _Y1, _X2, _Y2]
):
    def __init__(
        self,
        name: str,
        image_transformer: Transformer[_X1, _X2],
        data_transformer: Transformer[_Y1, _Y2],
    ):
        def f(img: _X1, data: _Y1) -> Tuple[_X2, _Y2]:
            return image_transformer(img), data_transformer(data)

        super().__init__(name, f)


class KeyedImageDatasetTransformer(
    Transformer[Tuple[_X1, _Y], Tuple[Union[_X1, _X2], _Y]],
    Generic[_X1, _Y, _X2],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., _X2],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
        key_getter: Callable[[_Y], Optional[str]] = operator.itemgetter(
            'name'
        ),
    ):
        self.packer = packer

        def g(
            img_data_pair: Tuple[_X1, _Y], *args, **kwargs
        ) -> Tuple[Union[_X1, _X2], _Y]:
            def mapped_f(img: _X1, data: _Y) -> Tuple[Union[_X1, _X2], _Y]:
                key = key_getter(data)
                img_f = self.get_image_transformer(f, key)
                return img_f(img), data

            return mapped_f(*img_data_pair)

        super().__init__(name, g)

    def get_image_transformer(
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

    def describe(self) -> JSONDataType:
        return funcy.merge(super().describe(), {'packer': self.packer})


class DictImageTransformer(
    FrozenNamedMixin,
    Mapping[str, Transformer[Tuple[_X1, _Y], Tuple[Union[_X1, _X2], _Y]]],
    Generic[_X1, _Y, _X2],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., _X2],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
    ):
        self.packer = packer
        self._transformer_mapping = KeyedDefaultDict(self._transformer_factory)
        self.func = f
        super().__init__(name)

    def _resolve_func_and_pack(
        self, key: str
    ) -> Tuple[Callable[[_X1], _X2], Pack]:
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
                raise ValueError(
                    f'Invalid key {key}: corresponding pack was not found.'
                    ' Define a default pack by passing None: default_pack'
                    ' or None: None to skip missing keys.'
                ) from e

        elif callable(self.packer):
            return self.func, self.packer(key)
        else:
            raise ValueError(
                'self.packer must be either a Mapping[Optional[str], Pack] '
                'or a Callable[[str], Pack]. '
                f'Got {type(self.packer)}'
            )

    def _transformer_factory(self, key: str) -> ImageTransformer[_X1, _Y, _X2]:
        name = '_'.join((self.name, key))
        func, pack = self._resolve_func_and_pack(key)
        return ImageTransformer(name, func, pack)

    def __iter__(self) -> Iterator[str]:
        return iter(self._transformer_mapping)

    def __len__(self) -> int:
        return len(self._transformer_mapping)

    def __getitem__(self, key: str):
        return self._transformer_mapping[key]

    def describe(self) -> JSONDataType:
        return {
            'type': self.__class__.__name__,
            'name': self.name,
            'packer': self.packer
            if isinstance(self.packer, Mapping)
            else self.packer.__name__,
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
