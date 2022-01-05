import enum
from typing import Dict, List, Optional, Type, TypeVar, Union

from nidaqmx.constants import ChannelType as NIChannelType
from nidaqmx.task import Task

from boiling_learning.daq.devices import Device
from boiling_learning.utils.utils import SimpleRepr, SimpleStr

T = TypeVar('T')

ChannelType = enum.Enum('ChannelType', 'UNDEFINED ANALOG COUNTER DIGITAL INPUT OUTPUT')


class Channel(SimpleRepr, SimpleStr):
    channel_table: Dict[str, str] = {}

    exclusive_types = [
        [
            ChannelType.UNDEFINED,
            ChannelType.ANALOG,
            ChannelType.COUNTER,
            ChannelType.DIGITAL,
        ],
        [ChannelType.UNDEFINED, ChannelType.INPUT, ChannelType.OUTPUT],
    ]

    channel_type_keys = {
        NIChannelType.ANALOG_INPUT: 'ai',
        NIChannelType.DIGITAL_INPUT: 'di',
        NIChannelType.COUNTER_INPUT: 'ci',
        NIChannelType.ANALOG_OUTPUT: 'ao',
        NIChannelType.DIGITAL_OUTPUT: 'do',
        NIChannelType.COUNTER_OUTPUT: 'co',
    }

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

    @property
    def path(self) -> str:
        return self.device.path + '/' + self.name

    def exists(self, task: Task) -> bool:
        return self.device.exists(task) and (
            self.path in task.channel_names or self.description in task.channel_names
        )

    def is_type(self, type1: ChannelType, type2: Optional[ChannelType] = None) -> bool:
        if type2 is None:
            return type1 in self.type
        else:
            return self.is_type(type1) and self.is_type(type2)

    def set_type(self, type1: ChannelType, type2: Optional[ChannelType] = None) -> None:
        if type2 is None:
            if type1 == ChannelType.UNDEFINED:
                self.type = (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
            else:
                for type_idx in range(len(self.exclusive_types)):
                    if type1 in self.exclusive_types[type_idx]:
                        self.type = tuple(
                            self.type[i] if i != type_idx else type1 for i in range(len(self.type))
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

        for ni_channel_type in NIChannelType:
            if all(t.name in ni_channel_type.name for t in self.type):
                return ni_channel_type

        return None

    def ni_type_key(self) -> Optional[str]:
        return Channel.channel_type_keys[self.ni_type] if self.ni_type is not None else None

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

    def call_ni(self, task: Task, method_name: str, *args, **kwargs):
        self.ni = getattr(getattr(task, self.ni_type_key() + '_channels'), method_name)(
            self.path, self.description, *args, **kwargs
        )
        return self.ni

    def add_to_task(self, task: Task, channel_specification: str, *args, **kwargs):
        self.call_ni(
            task,
            'add_' + self.ni_type_key() + '_' + channel_specification,
            *args,
            **kwargs,
        )
        self.add_to_task_table(task)
        return self.ni

    def read(
        self,
        task: Task,
        readings: Union[List[float], List[List[float]]],
        dtype: Optional[Type[T]] = None,
    ) -> Optional[T]:
        if dtype is None:
            dtype = list

        if len(Channel.channel_table[task.name]) > 0:
            if len(Channel.channel_table[task.name]) == 1:
                return dtype(readings[task.name])
            else:
                return dtype(readings[task.name][self.index_in_table(task)])
        else:
            return None
