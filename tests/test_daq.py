import unittest

import nidaqmx

from boiling_learning.daq import Channel, ChannelType, Device, NIChannelType

try:
    system = nidaqmx.system.System.local()
    set(device.name for device in system.devices)
except (nidaqmx.DaqError, nidaqmx._lib.DaqNotFoundError):
    pass
else:
    class daq_Device_test(unittest.TestCase):
        def test_init(self):
            d = Device()
            self.assertEqual(d.name, '')

            d = Device('Dev0')
            self.assertEqual(d.name, 'Dev0')

        def test_path(self):
            d = Device('Dev0')
            self.assertEqual(d.path, 'Dev0')

        def test_str(self):
            d = Device('Dev0')
            self.assertEqual(str(d), 'Device(name=Dev0)')

        def test_exists(self):
            d0 = Device('cDAQ1Mod4')
            d1 = Device('cDAQ1Mod6')
            d2 = Device('cDAQ1Mod100')

            system = nidaqmx.system.System.local()

            self.assertTrue(d0.exists(system))
            self.assertTrue(d1.exists(system))
            self.assertFalse(d2.exists(system))

            with nidaqmx.Task('Task') as task:
                task.ai_channels.add_ai_voltage_chan('cDAQ1Mod4/ai0')

                self.assertTrue(d0.exists(task))
                self.assertFalse(d1.exists(task))
                self.assertFalse(d2.exists(task))

        def test_equality(self):
            self.assertEqual(Device(), Device())
            self.assertEqual(Device('Dev0'), Device('Dev0'))


    class daq_Channel_test(unittest.TestCase):
        def test_init(self):
            ch = Channel(Device('Dev0'), name='ai0', description='My Channel')
            self.assertEqual(ch.name, 'ai0')
            self.assertEqual(ch.desc, 'My Channel')
            self.assertEqual(ch.device, Device('Dev0'))
            self.assertEqual(ch.type, (ChannelType.UNDEFINED, ChannelType.UNDEFINED))
            self.assertIsNone(ch.ni)

        def test_path(self):
            ch = Channel(Device('Dev0'), name='ai0')
            self.assertEqual(ch.path, 'Dev0/ai0')

        def test_description(self):
            ch = Channel(Device())
            self.assertEqual(ch.description, '')

            ch = Channel(Device(), description='My Channel')
            self.assertEqual(ch.description, 'My Channel')

        def test_exists(self):
            with nidaqmx.Task('Task') as task:
                task.ai_channels.add_ai_voltage_chan('cDAQ1Mod4/ai0')
                print(task.channel_names)

                self.assertTrue(Channel(Device('cDAQ1Mod4'), 'ai0').exists(task))
                self.assertFalse(Channel(Device('cDAQ1Mod100'), 'ai3').exists(task))
                self.assertFalse(Channel(Device('cDAQ1Mod4'), 'ai300').exists(task))

        def test_type(self):
            ch = Channel(Device())
            self.assertTrue(ch.is_type(ChannelType.UNDEFINED))
            self.assertTrue(ch.is_type(ChannelType.UNDEFINED, ChannelType.UNDEFINED))
            self.assertFalse(ch.is_type(ChannelType.UNDEFINED, ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.UNDEFINED))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.INPUT))

            ch = Channel(Device(), type1=ChannelType.ANALOG)
            self.assertTrue(ch.is_type(ChannelType.UNDEFINED))
            self.assertFalse(ch.is_type(ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.DIGITAL))
            self.assertFalse(ch.is_type(ChannelType.OUTPUT))
            self.assertTrue(ch.is_type(ChannelType.ANALOG))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.INPUT))

            ch = Channel(Device(), type1=ChannelType.ANALOG, type2=ChannelType.INPUT)
            self.assertFalse(ch.is_type(ChannelType.UNDEFINED))
            self.assertTrue(ch.is_type(ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.DIGITAL))
            self.assertFalse(ch.is_type(ChannelType.OUTPUT))
            self.assertTrue(ch.is_type(ChannelType.ANALOG))
            self.assertTrue(ch.is_type(ChannelType.ANALOG, ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.OUTPUT))

            ch = Channel(Device())
            ch.set_type(ChannelType.ANALOG)
            self.assertTrue(ch.is_type(ChannelType.UNDEFINED))
            self.assertFalse(ch.is_type(ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.DIGITAL))
            self.assertFalse(ch.is_type(ChannelType.OUTPUT))
            self.assertTrue(ch.is_type(ChannelType.ANALOG))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.INPUT))

            ch.set_type(ChannelType.INPUT)
            self.assertFalse(ch.is_type(ChannelType.UNDEFINED))
            self.assertTrue(ch.is_type(ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.DIGITAL))
            self.assertFalse(ch.is_type(ChannelType.OUTPUT))
            self.assertTrue(ch.is_type(ChannelType.ANALOG))
            self.assertTrue(ch.is_type(ChannelType.ANALOG, ChannelType.INPUT))
            self.assertFalse(ch.is_type(ChannelType.ANALOG, ChannelType.OUTPUT))

        def test_ni_type(self):
            ch = Channel(Device())
            self.assertIsNone(ch.ni_type())
            self.assertIsNone(ch.ni_type_key())

            ch = Channel(Device(), type1=ChannelType.ANALOG)
            self.assertIsNone(ch.ni_type())
            self.assertIsNone(ch.ni_type_key())

            ch = Channel(Device(), type1=ChannelType.ANALOG, type2=ChannelType.INPUT)
            self.assertEqual(ch.ni_type(), NIChannelType.ANALOG_INPUT)
            self.assertTrue(ch.is_ni_type(NIChannelType.ANALOG_INPUT))
            self.assertFalse(ch.is_ni_type(NIChannelType.ANALOG_OUTPUT))

            ch = Channel(Device(), type1=ChannelType.ANALOG, type2=ChannelType.INPUT)
            self.assertEqual(ch.ni_type(), NIChannelType.ANALOG_INPUT)
            self.assertTrue(ch.is_ni_type(NIChannelType.ANALOG_INPUT))
            self.assertFalse(ch.is_ni_type(NIChannelType.ANALOG_OUTPUT))
            self.assertEqual(ch.ni_type_key(), 'ai')
            self.assertNotEqual(ch.ni_type_key(), 'ao')

        def test_task_table(self):
            with nidaqmx.Task('Task1') as task1, nidaqmx.Task('Task2') as task2:
                ch0 = Channel(Device('Dev0'), 'ai0')
                ch1 = Channel(Device('Dev1'), 'ai3')

                ch0.add_to_task_table(task1)
                ch1.add_to_task_table(task1)
                ch1.add_to_task_table(task2)

                self.assertEqual(ch0.index_in_table(task1), 0)
                self.assertEqual(ch1.index_in_table(task1), 1)
                self.assertEqual(ch1.index_in_table(task2), 0)
                self.assertIsNone(ch0.index_in_table(task2))

        def test_add_to_task(self):
            with nidaqmx.Task('Task') as task:
                ch = Channel(Device('cDAQ1Mod4'), 'ai0', type1=ChannelType.INPUT, type2=ChannelType.ANALOG)
                ch.add_to_task(task, 'voltage_chan')

                self.assertTrue(ch.exists(task))
                self.assertEqual(ch.index_in_table(task), 0)
