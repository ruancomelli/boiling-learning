import nidaqmx
import pytest

from boiling_learning.daq.channels import Channel, ChannelType, NIChannelType
from boiling_learning.daq.devices import Device


def _has_nidaqmx() -> bool:
    try:
        system = nidaqmx.system.System.local()
        for device in system.devices:
            if device.name is None:
                return False
    except (nidaqmx.DaqError, nidaqmx._lib.DaqNotFoundError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_nidaqmx(), reason='no access to nidaqmx')


class Test_daq_Device:
    def test_init(self) -> None:
        d = Device()
        assert d.name == ''

        d = Device('Dev0')
        assert d.name == 'Dev0'

    def test_path(self) -> None:
        d = Device('Dev0')
        assert d.path == 'Dev0'

    def test_str(self) -> None:
        d = Device('Dev0')
        assert str(d) == 'Device(name=Dev0)'

    def test_exists(self) -> None:
        d0 = Device('cDAQ1Mod4')
        d1 = Device('cDAQ1Mod6')
        d2 = Device('cDAQ1Mod100')

        system = nidaqmx.system.System.local()

        assert d0.exists(system)
        assert d1.exists(system)
        assert not d2.exists(system)

        with nidaqmx.Task('Task') as task:
            task.ai_channels.add_ai_voltage_chan('cDAQ1Mod4/ai0')

            assert d0.exists(task)
            assert not d1.exists(task)
            assert not d2.exists(task)

    def test_equality(self) -> None:
        assert Device() == Device()
        assert Device('Dev0') == Device('Dev0')


class Test_daq_Channel:
    def test_init(self) -> None:
        ch = Channel(Device('Dev0'), name='ai0', description='My Channel')
        assert ch.name == 'ai0'
        assert ch.desc == 'My Channel'
        assert ch.device == Device('Dev0')
        assert ch.type == (ChannelType.UNDEFINED, ChannelType.UNDEFINED)
        assert ch.ni is None

    def test_path(self) -> None:
        ch = Channel(Device('Dev0'), name='ai0')
        assert ch.path == 'Dev0/ai0'

    def test_description(self) -> None:
        ch = Channel(Device())
        assert ch.description == ''

        ch = Channel(Device(), description='My Channel')
        assert ch.description == 'My Channel'

    def test_exists(self) -> None:
        with nidaqmx.Task('Task') as task:
            task.ai_channels.add_ai_voltage_chan('cDAQ1Mod4/ai0')
            print(task.channel_names)

            assert Channel(Device('cDAQ1Mod4'), 'ai0').exists(task)
            assert not Channel(Device('cDAQ1Mod100'), 'ai3').exists(task)
            assert not Channel(Device('cDAQ1Mod4'), 'ai300').exists(task)

    def test_ni_type(self) -> None:
        ch = Channel(Device())
        assert ch.ni_type is None
        assert ch.ni_type_key() is None

        ch = Channel(Device(), type1=ChannelType.ANALOG)
        assert ch.ni_type is None
        assert ch.ni_type_key() is None

        ch = Channel(Device(), type1=ChannelType.ANALOG, type2=ChannelType.INPUT)
        assert ch.ni_type == NIChannelType.ANALOG_INPUT
        assert ch.is_ni_type(NIChannelType.ANALOG_INPUT)
        assert not ch.is_ni_type(NIChannelType.ANALOG_OUTPUT)

        ch = Channel(Device(), type1=ChannelType.ANALOG, type2=ChannelType.INPUT)
        assert ch.ni_type == NIChannelType.ANALOG_INPUT
        assert ch.is_ni_type(NIChannelType.ANALOG_INPUT)
        assert not ch.is_ni_type(NIChannelType.ANALOG_OUTPUT)
        assert ch.ni_type_key() == 'ai'
        assert ch.ni_type_key() != 'ao'

    def test_task_table(self) -> None:
        with nidaqmx.Task('Task1') as task1, nidaqmx.Task('Task2') as task2:
            ch0 = Channel(Device('Dev0'), 'ai0')
            ch1 = Channel(Device('Dev1'), 'ai3')

            ch0.add_to_task_table(task1)
            ch1.add_to_task_table(task1)
            ch1.add_to_task_table(task2)

            assert ch0.index_in_table(task1) == 0
            assert ch1.index_in_table(task1) == 1
            assert ch1.index_in_table(task2) == 0
            assert ch0.index_in_table(task2) is None

    def test_add_to_task(self) -> None:
        with nidaqmx.Task('Task') as task:
            ch = Channel(
                Device('cDAQ1Mod4'),
                'ai0',
                type1=ChannelType.INPUT,
                type2=ChannelType.ANALOG,
            )
            ch.add_to_task(task, 'voltage_chan')

            assert ch.exists(task)
            assert ch.index_in_table(task) == 0
