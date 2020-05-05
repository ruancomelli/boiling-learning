import enum
from nidaqmx.constants import ChannelType as NIChannelType

ChannelType = enum.Enum('ChannelType', 'UNDEFINED ANALOG COUNTER DIGITAL INPUT OUTPUT')

class Channel:
    channel_table = {}

    exclusive_types = [
        [ChannelType.UNDEFINED, ChannelType.ANALOG, ChannelType.COUNTER, ChannelType.DIGITAL],
        [ChannelType.UNDEFINED, ChannelType.INPUT, ChannelType.OUTPUT]
    ]

    channel_type_keys = {
        NIChannelType.ANALOG_INPUT: 'ai',
        NIChannelType.DIGITAL_INPUT: 'di',
        NIChannelType.COUNTER_INPUT: 'ci',
        NIChannelType.ANALOG_OUTPUT: 'ao',
        NIChannelType.DIGITAL_OUTPUT: 'do',
        NIChannelType.COUNTER_OUTPUT: 'co'
    }

    def __init__(self, device, name='', description='', type1=ChannelType.UNDEFINED, type2=ChannelType.UNDEFINED):
        self.device = device
        self.name = name
        self.desc = description
        self.type = (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
        self.set_type(type1, type2)
        self.ni = None

    def __str__(self):
        return self.path()

    def path(self):
        return self.device.path() + '/' + self.name

    def exists(self, task):
        return self.device.exists(task) and (self.path() in task.channel_names or self.description() in task.channel_names)

    def description(self, desc=None):
        if desc is not None:
            self.desc = desc
        return self.desc

    def is_type(self, type1, type2=None):
        if type2 is None:
            return type1 in self.type
        else:
            return self.is_type(type1) and self.is_type(type2)

    def set_type(self, type1, type2=None):
        if type2 is None:
            if type1 == ChannelType.UNDEFINED:
                self.type = (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
            else:
                for type_idx in range(len(self.exclusive_types)):
                    if type1 in self.exclusive_types[type_idx]:
                        self.type = tuple(self.type[i] if i != type_idx else type1 for i in range(len(self.type)))

        else:
            if type2 != ChannelType.UNDEFINED:
                self.set_type(type1)
                self.set_type(type2)
            else:
                self.set_type(type2)
                self.set_type(type1)

    def ni_type(self):
        if ChannelType.UNDEFINED not in self.type:
            for ni_channel_type in NIChannelType:
                if all([t.name in ni_channel_type.name for t in self.type]):
                    return ni_channel_type
        else:
            return None

    def is_ni_type(self, ni_channel_type):
        return self.ni_type() is ni_channel_type

    def ni_type_key(self):
        return Channel.channel_type_keys[self.ni_type()] if self.ni_type() is not None else None

    def add_to_task_table(self, task):
        if task.name not in Channel.channel_table:
            Channel.channel_table[task.name] = []
        Channel.channel_table[task.name].append(self.path())

    def index_in_table(self, task):
        return Channel.channel_table[task.name].index(self.path()) if self.path() in Channel.channel_table[task.name] else None

    def call_ni(self, task, method_name, *args, **kwargs):
        self.ni = getattr(getattr(task, self.ni_type_key() + '_channels'), method_name)(
                self.path(), self.description(),
                *args, **kwargs)
        return self.ni

    def add_to_task(self, task, channel_specification, *args, **kwargs):
        self.call_ni(task, 'add_' + self.ni_type_key() + '_' + channel_specification, *args, **kwargs)
        self.add_to_task_table(task)
        return self.ni

    def read(self, task, readings, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = list

        if len(Channel.channel_table[task.name]) > 0:
            if len(Channel.channel_table[task.name]) == 1:
                return dtype(readings[task.name])
            else:
                return dtype(readings[task.name][self.index_in_table(task)])
        else:
            return None
