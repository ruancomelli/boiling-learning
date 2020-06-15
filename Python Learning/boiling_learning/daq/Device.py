from nidaqmx.task import Task

import boiling_learning as bl

class Device(bl.utils.SimpleRepr, bl.utils.SimpleStr, bl.utils.DictEq):
    def __init__(self, name: str = ''):
        self.name = name

    @property
    def path(self) -> str:
        return self.name

    def exists(self, task: Task) -> bool:
        return self.path in set(device.name for device in task.devices)