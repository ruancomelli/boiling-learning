# -*- coding: utf-8 -*-
"""
@author: ruan.comelli
"""
#%%
import os
import pathlib

def add_to_system_path(path_to_add, add_if_exists=False):
    str_to_add = str(path_to_add)
    if add_if_exists or str_to_add not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + str(str_to_add)

# ensure that anaconda is in system's PATH
if os.environ['COMPUTERNAME'] == 'LABSOLAR29-001':
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'mingw-w64' / 'bin')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'usr' / 'bin')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'bin')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Scripts')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'bin')
    add_to_system_path(pathlib.Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'condabin')

import nidaqmx
from datetime import datetime
import numpy as np
import csv
import pandas as pd
import time

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from Channel import ChannelType, Channel
from LivePlotter import LivePlotter
from Device import Device

#%%
"""
Settings -------------------------------------------------------
"""

# def serial():
#     import serial
#     import datetime

#     serial_path = 'COM3'
#     baud_rate   = 9600

#     ser = serial.Serial(serial_path, baud_rate)

#     with open("output.txt",'x') as file:
#         try:
#             while True:
#                 output = []
#                 while len(output) == 0:
#                     output = ser.read_until(b"ENTER.")
#                 output = str(output)
#                 now    = datetime.now()
#                 to_file = str(now)+';'+str(output)
#                 file.write(to_file+'\n')
#                 print(to_file)
#         except (OSError, serial.SerialException):
#             ser.close()

read_continuously = True
maximum_number_of_iterations = 5

must_print = {
    'anything': True, # Use false to disallow printing at all
    'info': True,
    'voltage': True,
    'current': True,
    'power': True,
    'temperature': True,
    'writing': True,
    'led read state': True,
    'elapsed time': True,
    'sleeping': False
}

sample_rate = 4 # Hz
sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
samples_per_channel = 1000000
sample_period = 1 / sample_rate # s
sleeping_time = 0.5 * sample_period

write_period = 10

calibration_filepath = pathlib.Path(__file__).parent.parent / 'Experimental Set Calibration' / 'Processing' / 'coefficients.csv'
output_dir_pattern = str(pathlib.Path() / r'Experiment Output %d-%m-%Y')
filename_pattern = r'Experiment {index} -- %H-%M.csv'

figsize = (13, 6)
x_axis_size = 100 * sample_rate

should_plot = True

# For timestamps: <https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z000000kJy2SAE&l=pt-BR>

#%%
""" 
Support definitions -------------------------------------------------------
"""
def format_output_dir(output_dir_pattern):
    return pathlib.Path(datetime.now().strftime(output_dir_pattern))

def format_filename(output_dir, file_pattern):
    format_time = lambda fpattern, counter: datetime.now().strftime(fpattern.format(index=counter))
    substituted_filepath = lambda fpattern, counter: output_dir / format_time(fpattern, counter)

    counter = 0
    while substituted_filepath(file_pattern, counter).is_file():
        counter += 1
    return format_time(file_pattern, counter)

def generate_empty_copy(local_data):
    return dict.fromkeys(local_data, np.array([]))

def print_if(cond, *args, **kwargs):
    if cond:
        print(*args, **kwargs)
        
def print_if_must(keys, *args, conds=(), **kwargs):
    print_if(all(must_print[key] for key in keys) and all(conds), *args, **kwargs)
    
#%%
"""
Channel definitions -------------------------------------------------------
"""
voltage_channels = {
    'Orange Resistor': Channel(Device('cDAQ1Mod4'), 'ai3', 'Orange Resistor', ChannelType.ANALOG, ChannelType.INPUT),
    'Blue Resistor': Channel(Device('cDAQ1Mod4'), 'ai4', 'Blue Resistor', ChannelType.ANALOG, ChannelType.INPUT),
    'Yellow Resistor': Channel(Device('cDAQ1Mod4'), 'ai5', 'Yellow Resistor', ChannelType.ANALOG, ChannelType.INPUT)
}
current_channel = Channel(Device('cDAQ1Mod4'), 'ai0', 'Current Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)
rtd_channel = Channel(Device('cDAQ1Mod6'), 'ai0', 'RTD Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)
led_reading_channel = Channel(Device('cDAQ1Mod4'), 'ai7', 'LED Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)

#%%
""" 
Print system information -------------------------------------------------------
"""
system = nidaqmx.system.System.local()
print_if_must(('anything', 'info'), f'> {system.driver_version}')
print_if_must(('anything', 'info'), f'> Available devices in system.devices: {[x for x in system.devices]}')
print_if_must(('anything', 'info'), f'> AI channels in system.devices: { {x: [y.name for y in x.ai_physical_chans] for x in system.devices} }')
print_if_must(('anything', 'info'), f'> DO ports in system.devices: { {x: [y.name for y in x.do_ports] for x in system.devices} }')
print_if_must(('anything', 'info'), f'> Types in ChannelType: {[x for x in ChannelType]}')

#%%
""" 
Load calibration polynomial -------------------------------------------------------
"""
with open(calibration_filepath, 'r') as calibration_coefficients_file:
    coefficients = calibration_coefficients_file.readlines()
    coefficients = [float(x.strip()) for x in coefficients]
    calibrated_polynomial = np.poly1d(list(reversed(coefficients)))
    print_if_must(('anything', 'info'), f'> Calibrated polynomial:\n{calibrated_polynomial}')

#%%
"""
Initialize -------------------------------------------------------
"""
output_dir = format_output_dir(output_dir_pattern)
filepath = output_dir / format_filename(output_dir, filename_pattern)
os.makedirs(output_dir, exist_ok=True)
print_if_must(('anything', 'info'), f'> File path: {filepath}')


if should_plot:
    ### START QtApp #####
    app = QtGui.QApplication([]) # see <https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph>
    ####################

with open(filepath, 'w', newline='') as output_file, \
     nidaqmx.Task('Experiment') as experiment:

    output_writer = csv.writer(output_file)

#%%
    """
    Setup channels -------------------------------------------------------
    """
    for channel_nickname, voltage_reading_channel in voltage_channels.items():
        voltage_channels[channel_nickname].add_to_task(
            task=experiment, channel_specification='voltage_chan',
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=0.0, max_val=10.0)

    rtd_channel.add_to_task(
        task=experiment, channel_specification='rtd_chan',
        resistance_config=nidaqmx.constants.ResistanceConfiguration.FOUR_WIRE,
        min_val=0.0, max_val=100.0,
        rtd_type=nidaqmx.constants.RTDType.CUSTOM,
        current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL, current_excit_val=1e-3,
        r_0=100)
    rtd_channel.ni.ai_rtd_a = 3.9083e-3 # This is how the original rtd was defined for the calibration
    rtd_channel.ni.ai_rtd_b = -577.5e-9
    rtd_channel.ni.ai_rtd_c = -4.183e-12

    current_channel.add_to_task(
        task=experiment, channel_specification='current_chan',
        terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
        min_val=-15, max_val=15,
        shunt_resistor_loc=nidaqmx.constants.CurrentShuntResistorLocation.EXTERNAL,
        ext_shunt_resistor_val=4e-3
    )
    led_reading_channel.add_to_task(
        task=experiment, channel_specification='voltage_chan',
        terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
        min_val=0.0, max_val=7.0
    )
    
    experiment.timing.cfg_samp_clk_timing(sample_rate, sample_mode=sample_mode)
    experiment.start()

    print_if_must(('anything', 'info'), f'experiment samp_clk_rate: {experiment.timing.samp_clk_rate}')

#%%
    """
    Run experiment -------------------------------------------------------
    """
    first = True
    should_continue = True
    iter_count = 0
    readings = {}
    start = time.time()
    while first or should_continue:
        print_if_must(('anything', 'info'), f'> Iteration {iter_count}')
#%%
        """
        Read data -------------------------------------------------------
        """
        readings[experiment.name] = experiment.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        elapsed_time = np.array([time.time() - start])
#%%
        """
        Process data -------------------------------------------------------
        """
        # Electric data:
        voltage_readings = {key: chan.read(experiment, readings, dtype=np.array) for key, chan in voltage_channels.items()}
        voltage_readings_values = np.array(list(voltage_readings.values()))
        voltage = np.sum(voltage_readings_values, axis=0) if voltage_readings_values.size > 0 else voltage_readings_values
        current = current_channel.read(experiment, readings, dtype=np.array)
        power = voltage * current
        
        # Thermal data:
        rtd_read_value = rtd_channel.read(experiment, readings, dtype=np.array)
        rtd_temperature = rtd_read_value
        if rtd_temperature.size > 0:
            rtd_temperature = calibrated_polynomial(rtd_temperature)
            
        # LED data:
        led_voltage = led_reading_channel.read(experiment, readings, dtype=np.array)

#%%
        """
        Saving -------------------------------------------------------
        """        
        local_data = {
            'Time instant': np.array([datetime.fromtimestamp(start + et) for et in elapsed_time]),
            'Elapsed time': elapsed_time,
            'Voltage': voltage,
            'Current': current,
            'Power': power,
            'Temperature': rtd_temperature,
            'LED voltage': led_voltage
        }
        keys = local_data.keys()
        n_values = min([local.size for local in local_data.values()])
        
        if n_values > 0:
            elapsed_time += np.arange(n_values)*sample_period
            print_if_must(('anything', 'elapsed time'), f'Elapsed time [s]: {elapsed_time}')

        if first:
            data = generate_empty_copy(local_data)

        for key in keys:
            data[key] = np.append(data[key], local_data[key])
        
        continue_key = False
        for key, a in local_data.items():
            if a.size == 0:
                print_if_must(('anything', 'sleeping'), f'{key} is causing sleep')
                continue_key = True
        if continue_key:
            time.sleep(sleeping_time)
            continue

        # if any(a.size == 0 for a in local_data.values()):
        #     time.sleep(sleeping_time)
        #     continue

#%%
        """
        Writing to file -------------------------------------------------------
        """        
        if first:
            output_writer.writerow(keys)
        
        if iter_count % write_period == 0:
            print_if_must(('anything', 'writing'), '>> Writing to file...', end='')

            output_writer.writerows(zip(*[data[key] for key in keys]))
            data = generate_empty_copy(local_data)

            print_if_must(('anything', 'writing'), '>> Done')

#%%
        """  
        Printing -------------------------------------------------------
        """
        print_if_must(['anything'], '>>')
        print_if_must(('anything', 'voltage'), f'>> Voltage [V]: {voltage}', conds=[voltage.size > 0])
        print_if_must(('anything', 'current'), f'>> Current [A]: {current}', conds=[current.size > 0])
        print_if_must(('anything', 'power'), f'>> Power [W]: {power}', conds=[power.size > 0])
        print_if_must(('anything', 'temperature'), f'>> Temperature [°C]: {rtd_temperature}', conds=[rtd_temperature.size > 0])
        print_if_must(('anything', 'led read state'), f'>> LED: {led_voltage}', conds=[led_voltage.size > 0])

#%%
        """
        Plotting -------------------------------------------------------
        """
        if should_plot:
            number_of_variables = len(local_data)

            if first:
                win = pg.GraphicsWindow(title=filepath.name)
                ps = {
                    'Voltage': win.addPlot(title='Voltage', row=0, col=4),
                    'Current': win.addPlot(title='Current', row=1, col=4),
                    'Power': win.addPlot(title='Power', row=2, col=4),
                    'Temperature': win.addPlot(title='Temperature', row=0, col=0, rowspan=2, colspan=3),
                    'LED voltage': win.addPlot(title='LED Voltage', row=2, col=0, colspan=3)
                }
                # ps = {key: win.addPlot(title=key) for key in keys}
                curves = {key: ps[key].plot() for key in ps}

                x = {key: np.zeros(x_axis_size) for key in ps}
                y = {key: np.zeros(x_axis_size) for key in ps}

            for key in ps:                
                x[key][0:n_values] = elapsed_time
                x[key] = np.roll(x[key], -n_values)
                
                y[key][0:n_values] = local_data[key]
                y[key] = np.roll(y[key], -n_values)

                if iter_count < x_axis_size:
                    curves[key].setData(x[key][x_axis_size-iter_count+1:], y[key][x_axis_size-iter_count+1:])
                    curves[key].setPos(0, 0)
                else:
                    curves[key].setData(x[key], y[key])
                    curves[key].setPos(iter_count - x_axis_size, 0)
                    
            QtGui.QApplication.processEvents()

#%%
        """
        Decide -------------------------------------------------------
        """
        iter_count += 1
        first = False
        if not read_continuously:
            should_continue = iter_count <= maximum_number_of_iterations


if should_plot:
    ### END QtApp ####
    pg.QtGui.QApplication.exec_() # you MUST put this at the end
    ##################
    
