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

from boiling_learning.utils.dtypes import auto_spec, new_py_function
from boiling_learning.utils.functional import Pack, nth_arg
from boiling_learning.utils.utils import (
    FrozenNamedMixin,
    JSONDataType,
    KeyedDefaultDict,
    SimpleStr,
)

# from boiling_learning.io.json_encoders import (
#     PackEncoder
# )

C = TypeVar('C')
T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')
V = TypeVar('V')


class Transformer(FrozenNamedMixin, SimpleStr, Generic[T, S]):
    def __init__(self, name: str, f: Callable[..., S], pack: Pack = Pack()):
        super().__init__(name)

        self.pack = pack
        self.transformer = pack.rpartial(f)

    def __call__(self, arg: T, *args, **kwargs) -> S:
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
            return new_py_function(func=func, inp=args, Tout=auto_spec(args))

        return _tf_py_function


class Creator(Transformer[Pack, S], Generic[S]):
    def __init__(
        self,
        name: str,
        f: Callable[..., S],
        pack: Pack = Pack(),
        expand_pack_on_call: bool = False,
    ):
        if expand_pack_on_call:

            def g(pack: Pack, *args, **kwargs) -> S:
                return f(*(pack.args + args), **{**pack.kwargs, **kwargs})

        else:
            g = f

        super().__init__(name, g, pack=pack)


class ImageTransformer(
    Transformer[Tuple[T, U], Tuple[S, U]], Generic[T, U, S]
):
    def __init__(self, name: str, f: Callable[..., S], pack: Pack = Pack()):
        def g(img_data_pair: Tuple[T, U], *args, **kwargs) -> Tuple[S, U]:
            def pair_transformer(img: T, data: U) -> Tuple[S, U]:
                return f(img, *args, **kwargs), data

            return pair_transformer(*img_data_pair)

        super().__init__(name, g, pack=pack)

    def transform_image(self, img: T, *args, **kwargs) -> S:
        return self((img, None), *args, **kwargs)[0]

    def as_image_transformer(self) -> Transformer[T, S]:
        return Transformer(
            '_'.join((self.name, 'image_function')), self.transform_image
        )


class ImageDatasetTransformer(
    Transformer[Tuple[T, U], Tuple[S, V]], Generic[T, U, S, V]
):
    def __init__(
        self,
        name: str,
        image_transformer: Transformer[T, S],
        data_transformer: Transformer[U, V],
    ):
        def f(img: T, data: U) -> Tuple[S, V]:
            return image_transformer(img), data_transformer(data)

        super().__init__(name, f)


class KeyedImageDatasetTransformer(
    Transformer[Tuple[T, U], Tuple[Union[T, S], U]], Generic[T, U, S]
):
    def __init__(
        self,
        name: str,
        f: Callable[..., S],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
        key_getter: Callable[[U], Optional[str]] = operator.itemgetter('name'),
    ):
        self.packer = packer

        def g(img_data_pair: Tuple[T, U], *args, **kwargs) -> Tuple[S, U]:
            def mapped_f(img: T, data: U) -> Tuple[Union[T, S], U]:
                key = key_getter(data)
                img_f = self.get_image_transformer(f, key)
                return img_f(img), data

            return mapped_f(*img_data_pair)

        super().__init__(name, g)

    def get_image_transformer(
        self, f: Callable[..., S], key: Optional[str]
    ) -> Callable[[T], Union[T, S]]:
        if callable(self.packer):
            pack = self.packer(key)
            return pack.rpartial(f)

        if key in self.packer:
            return self._get_partial_transformer(f, key)

        if None in self.packer:
            return self._get_partial_transformer(f, None)

        return funcy.identity

    def _get_partial_transformer(
        self, f: Callable[..., S], key: Optional[str]
    ) -> Callable[[T], S]:
        return self.packer[key].rpartial(f)

    def describe(self) -> JSONDataType:
        return funcy.merge(super().describe(), {'packer': self.packer})


class DictImageTransformer(
    FrozenNamedMixin,
    Mapping[
        str,
        Transformer[Tuple[T, U], Tuple[Union[T, S], U]],
    ],
    Generic[T, U, S],
):
    def __init__(
        self,
        name: str,
        f: Callable[..., S],
        packer: Union[Callable[[str], Pack], Mapping[Optional[str], Pack]],
    ):
        self.packer = packer
        self._transformer_mapping = KeyedDefaultDict(self._transformer_factory)
        self.func = f
        super().__init__(name)

    def _resolve_func_and_pack(
        self, key: str
    ) -> Tuple[Callable[[T], S], Pack]:
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

    def _transformer_factory(self, key: str) -> ImageTransformer[T, U, S]:
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


# class PackTransformerEncoder(PackEncoder):
#     def default(self, obj):
#         if isinstance(obj, Transformer):
#             return obj.describe()
#         # Let the base class default method raise the TypeError
#         return PackEncoder.default(self, obj)


first_argument_transformer = Transformer('first_argument', nth_arg(0))
