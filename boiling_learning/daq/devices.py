from nidaqmx.task import Task

from boiling_learning.utils.utils import DictEq, SimpleRepr, SimpleStr


class Device(SimpleRepr, SimpleStr, DictEq):
    def __init__(self, name: str = '') -> None:
        self.name = name

    @property
    def path(self) -> str:
        return self.name

    def exists(self, task: Task) -> bool:
        return self.path in {device.name for device in task.devices}
