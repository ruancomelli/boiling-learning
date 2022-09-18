from nidaqmx.task import Task

from boiling_learning.io.storage import dataclass


@dataclass
class Device:
    name: str = ''

    @property
    def path(self) -> str:
        return self.name

    def exists(self, task: Task) -> bool:
        return self.path in {device.name for device in task.devices}
