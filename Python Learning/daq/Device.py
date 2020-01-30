class Device:
    def __init__(self, name=''):
        self.name = name

    def path(self):
        return self.name

    def exists(self, task):
        return self.path() in [device.name for device in task.devices]

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        elif isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return NotImplemented