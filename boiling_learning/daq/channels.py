import enum
from typing import Optional, Type, TypeVar, Union

from frozendict import frozendict
from nidaqmx.constants import ChannelType as NIChannelType
from nidaqmx.task import Task

from boiling_learning.daq.devices import Device

T = TypeVar('T')


class ChannelType(enum.Enum):
    UNDEFINED = enum.auto()
    ANALOG = enum.auto()
    COUNTER = enum.auto()
    DIGITAL = enum.auto()
    INPUT = enum.auto()
    OUTPUT = enum.auto()


EXCLUSIVE_GROUPS = (
    frozenset(
        {ChannelType.UNDEFINED, ChannelType.ANALOG, ChannelType.COUNTER, ChannelType.DIGITAL}
    ),
    frozenset({ChannelType.UNDEFINED, ChannelType.INPUT, ChannelType.OUTPUT}),
)

CHANNEL_TYPE_KEYS = frozendict(
    {
        NIChannelType.ANALOG_INPUT: 'ai',
        NIChannelType.DIGITAL_INPUT: 'di',
        NIChannelType.COUNTER_INPUT: 'ci',
        NIChannelType.ANALOG_OUTPUT: 'ao',
        NIChannelType.DIGITAL_OUTPUT: 'do',
        NIChannelType.COUNTER_OUTPUT: 'co',
    }
)


class Channel:
    channel_table: dict[str, list[str]] = {}

    def __init__(
        self,
        device: Device,
        name: str = '',
        description: str = '',
        type1: ChannelType = ChannelType.UNDEFINED,
        type2: ChannelType = ChannelType.UNDEFINED,
    ) -> None:
        self.device = device
        self.name = name
        self.description = description
        self.type = (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
        self.set_type(type1, type2)
        self.ni = None

    def __repr__(self) -> str:
        # Source: <https://stackoverflow.com/a/44595303/5811400>
        class_name = self.__class__.__name__
        address = id(self) & 0xFFFFFF
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())

        return f'<{class_name} @{address:x} {attrs}>'

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        attrs = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())

        return f'{class_name}({attrs})'

    @property
    def path(self) -> str:
        return f'{self.device.path}/{self.name}'

    def exists(self, task: Task) -> bool:
        return self.device.exists(task) and (
            self.path in task.channel_names or self.description in task.channel_names
        )

    def set_type(self, type1: ChannelType, type2: Optional[ChannelType] = None) -> None:
        if type2 is None:
            if type1 == ChannelType.UNDEFINED:
                self.type = (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
            else:
                for type_idx, exclusive_group in enumerate(EXCLUSIVE_GROUPS):
                    if type1 in exclusive_group:
                        self.type = tuple(
                            type_ if i != type_idx else type1 for i, type_ in enumerate(self.type)
                        )

        elif type2 != ChannelType.UNDEFINED:
            self.set_type(type1)
            self.set_type(type2)
        else:
            self.set_type(type2)
            self.set_type(type1)

    @property
    def ni_type(self) -> Optional[NIChannelType]:
        if ChannelType.UNDEFINED in self.type:
            return None

        return next(
            (
                ni_channel_type
                for ni_channel_type in NIChannelType
                if all(t.name in ni_channel_type.name for t in self.type)
            ),
            None,
        )

    def ni_type_key(self) -> Optional[str]:
        return CHANNEL_TYPE_KEYS[self.ni_type] if self.ni_type is not None else None

    def add_to_task_table(self, task: Task) -> None:
        if task.name not in Channel.channel_table:
            Channel.channel_table[task.name] = []
        Channel.channel_table[task.name].append(self.path)

    def index_in_table(self, task: Task) -> Optional[int]:
        return (
            Channel.channel_table[task.name].index(self.path)
            if self.path in Channel.channel_table[task.name]
            else None
        )

    def add_to_task(self, task: Task, channel_specification: str, *args, **kwargs):
        ni_type_key = self.ni_type_key()
        method_name = f'add_{ni_type_key}_{channel_specification}'
        channels = getattr(task, f'{ni_type_key}_channels')
        method = getattr(channels, method_name)
        self.ni = method(self.path, self.description, *args, **kwargs)

        self.add_to_task_table(task)
        return self.ni

    def read(
        self,
        task: Task,
        readings: Union[list[float], list[list[float]]],
        dtype: Type[T] = list,
    ) -> Optional[T]:
        if not Channel.channel_table[task.name]:
            return None

        return (
            dtype(readings[task.name])
            if len(Channel.channel_table[task.name]) == 1
            else dtype(readings[task.name][self.index_in_table(task)])
        )
