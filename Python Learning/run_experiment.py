import os
from pathlib import Path

def add_to_system_path(path_to_add, add_if_exists=False):
    str_to_add = str(path_to_add)
    if add_if_exists or (str_to_add not in os.environ['PATH']):
        os.environ['PATH'] += os.pathsep + str_to_add

python_project_home_path = Path().absolute().resolve()
project_home_path = python_project_home_path.parent.resolve()

# ensure that anaconda is in system's PATH
# if os.environ['COMPUTERNAME'] == 'LABSOLAR29-001':
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'mingw-w64' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'usr' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Library' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'Scripts')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'bin')
#     add_to_system_path(Path('C:') / 'Users'/ 'ruan.comelli'/ 'AppData' / 'Local' / 'Continuum' / 'anaconda3' / 'condabin')


# defaultdict is a very useful class!!!

import nidaqmx
from datetime import datetime
import scipy
import numpy as np
import csv
import pandas as pd
import time
import functools

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

from daq import Channel, ChannelType, Device

class BoilingSurface:
    def __init__(self, settings):
        self.name = settings.get('name', None)
        self.name = self.name if self.name else None
        
        self.type = settings['type']
        
        if self.type == 'ribbon':
            self.length = settings['length']
            self.width = settings['width']
            self.thickness = settings['thickness']
        elif self.type == 'wire':
            self.length = settings['length']
            self.diameter = settings['diameter']
        else:
            raise ValueError(f'type {self.type} not supported')
        
    @property
    def cross_section_area(self):
        if self.type == 'ribbon':
            return self.width * self.thickness)
        elif self.type == 'wire':
            return np.pi * 0.25 * self.diameter**2
        else:
            raise ValueError(f'it is not possible to calculate cross section area for surface type {self.type}')
    
    @property
    def surface_area(self):
        if self.type == 'ribbon':
            return 2*self.length*(self.width + self.thickness)
        elif self.type == 'wire':
            return np.pi * self.length * self.diameter
        else:
            raise ValueError(f'it is not possible to calculate surface area for surface type {self.type}')
        

#%%
# -------------------------------------------------------
# Settings
# -------------------------------------------------------
def read_settings():
    import json
    
    with open(python_project_home_path / 'experiment_settings.json') as json_file:
        return json.load(json_file)

settings = read_settings()
surface = BoilingSurface(settings['surface'])

# ribbon or wire?
wire_diameter = 0.518e-3 # m
wire_length = 6.5e-2 # m
# wire_cross_section = np.pi * 0.25 * wire_diameter**2
# wire_surface_area = np.pi * wire_diameter * wire_length
wire_surface_area = surface.surface_area

read_continuously = True
maximum_number_of_iterations = 21

must_print = {
    'anything': True, # Use false to disallow printing at all
    'info': False,
    'voltage': False,
    'current': False,
    'power': False,
    'flux': True,
    'resistance': False,
    'bulk temperature': True,
    'writing': True,
    'led read state': False,
    'elapsed time': False,
    'sleeping': False,
    'wire temperature': True,
    'temperature from resistance': True,
}

sample_rate = 10 # Hz
sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
samples_per_channel = 100
sample_period = 1 / sample_rate # s
sleeping_time = 0.5 * sample_period

write_period = 100
calibration_filepath = project_home_path / 'Experimental Set Calibration' / 'Processing' / 'coefficients.csv'

# output_dir_pattern = str(python_project_home_path / 'experiments' / r'Experiment %Y-%m-%d %H-%M ({index})')
output_dir_pattern = str(python_project_home_path / 'experiments' / r'Experiment %Y-%m-%d %H-%M{optional_index}')
def optional_index_format(counter)
    return f' ({counter})' if counter != 0 else ''
# filename_pattern = r'Experiment %H-%M ({index}).csv'
filename_pattern = r'data.csv'

x_axis_size = 10 * sample_rate

should_plot = True

measure_loop_time = False
save = False

# Pairs (T, f) where T is the measured temperature in °C and f is the factor to multiply the wire resistance
factor_table = pd.DataFrame({
    'Temperature': [
        20,
        93,
        204,
        315,
        427,
        538,
        649,
        760,
        871,
        982,
        1093,
    ],
    'Factor': [
        1.000,
        1.016,
        1.037,
        1.054,
        1.066,
        1.070,
        1.064,
        1.062,
        1.066,
        1.072,
        1.078,
    ]
})

reference_temperature = 20 # deg C
reference_resistivity = (
    650
    * 1.66242611301008E-09 # ohm per circular mil-foot -> ohm meter
)
reference_resistance = reference_resistivity * wire_length / wire_cross_section

def calculate_resistance(voltage, current):
    return np.where(np.abs(current) >= 1e-8, np.abs(voltage / current), reference_resistance)

def calculate_temperature(resistance):
    from scipy import interpolate

    T_to_f = interpolate.interp1d(factor_table['Temperature'], factor_table['Factor'], copy=False, fill_value='extrapolate')
    reference_factor = T_to_f(reference_temperature)

    factor = resistance / reference_resistance
    f_to_T = interpolate.interp1d(factor_table['Factor'], factor_table['Temperature'], copy=False, fill_value='extrapolate')
    return f_to_T(factor)

@functools.lru_cache(maxsize=128)
def correct_wire_temperature(reference_file):
    df = pd.read_csv(reference_file, usecols=['Bulk Temperature [deg C]', 'Wire Temperature [deg C]'])
    mean_bulk_temperature = df['Bulk Temperature [deg C]'].mean()
    mean_wire_temperature = df['Wire Temperature [deg C]'].mean()

    return mean_bulk_temperature - mean_wire_temperature
wire_temperature_correction_reference_file = Path() / 'experiments' / 'Experiment Output 2020-02-14' / 'Experiment 10-03 (0).csv'
# wire_temperature_correction_reference_file = Path() / 'experiments' / 'Experiment Output 2020-02-18' / 'Experiment 16-44 (0).csv'
wire_temperature_correction = correct_wire_temperature(wire_temperature_correction_reference_file)

# For timestamps: <https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z000000kJy2SAE&l=pt-BR>

#%%
"""
Support definitions -------------------------------------------------------
"""
def format_output_dir(output_dir_pattern):
    from pathlib import Path

    full_dir_pattern = str(Path() / 'experiments' / datetime.now().strftime(output_dir_pattern))

    def format_time(dirpattern, counter): 
        return datetime.now().strftime(
            dirpattern.format(
                index=counter,
                optional_index=optional_index_format(counter)
            )
        )
    def substituted_output_dir(dirpattern, counter): 
        return Path(format_time(full_dir_pattern, counter))

    counter = 0
    if any(key in output_dir_pattern for key in ('{index}', '{optional_index}')):
        while substituted_output_dir(full_dir_pattern, counter).is_dir():
            counter += 1
        
    return format_time(full_dir_pattern, counter)

def format_filename(output_dir, file_pattern):
    def format_time(fpattern, counter):
        return datetime.now().strftime(fpattern.format(index=counter))
    def substituted_filepath(fpattern, counter):
        return output_dir / format_time(fpattern, counter)

    counter = 0
    if '{index}' in file_pattern:
        while substituted_filepath(file_pattern, counter).is_file():
            counter += 1
    return format_time(file_pattern, counter)

def generate_empty_copy(local_data):
    return dict.fromkeys(local_data, np.array([]))

def print_if(cond, *args, **kwargs):
    if cond:
        print(*args, **kwargs)

def print_if_must(keys, *args, conds=None, **kwargs):
    if conds is None:
        conds = []

    cond = all(must_print.get(key, False) for key in keys) and all(conds)
    print_if(cond, *args, **kwargs)

#%%
# -------------------------------------------------------
# Channel definitions
# -------------------------------------------------------
voltage_channels = {
    'Orange Resistor': Channel(Device('cDAQ1Mod4'), 'ai3', 'Orange Resistor', ChannelType.ANALOG, ChannelType.INPUT),
    'Blue Resistor': Channel(Device('cDAQ1Mod4'), 'ai4', 'Blue Resistor', ChannelType.ANALOG, ChannelType.INPUT),
    'Yellow Resistor': Channel(Device('cDAQ1Mod4'), 'ai5', 'Yellow Resistor', ChannelType.ANALOG, ChannelType.INPUT)
}
# voltage_channels = {
#     'Wire Voltage': Channel(Device('cDAQ1Mod4'), 'ai2', 'Wire Voltage', ChannelType.ANALOG, ChannelType.INPUT)
# }
current_channel = Channel(Device('cDAQ1Mod4'), 'ai0', 'Current Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)
rtd_channel = Channel(Device('cDAQ1Mod6'), 'ai0', 'RTD Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)
led_reading_channel = Channel(Device('cDAQ1Mod4'), 'ai7', 'LED Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)
thermocouple_channel = Channel(Device('cDAQ1Mod1'), 'ai1', 'Thermocouple Reading Channel', ChannelType.ANALOG, ChannelType.INPUT)

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
# -------------------------------------------------------
# Initialize
# -------------------------------------------------------
output_dir = Path(format_output_dir(output_dir_pattern))
output_dir.mkdir(parents=True, exist_ok=True)
filepath = output_dir / format_filename(output_dir, filename_pattern)
print_if_must(('anything', 'info'), f'> File path: {filepath}')


if should_plot:
    ### START QtApp #####
    app = QtGui.QApplication([]) # see <https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph>
    ####################

with open(filepath, 'w', newline='') as output_file, \
     nidaqmx.Task('Experiment') as experiment:

    output_writer = csv.writer(output_file)

#%%
    # -------------------------------------------------------
    # Setup channels
    # -------------------------------------------------------
    for channel_nickname, voltage_reading_channel in voltage_channels.items():
        voltage_channels[channel_nickname].add_to_task(
            task=experiment, channel_specification='voltage_chan',
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=0.0, max_val=10.0)

    rtd_channel.add_to_task(
        task=experiment, channel_specification='rtd_chan',
        resistance_config=nidaqmx.constants.ResistanceConfiguration.FOUR_WIRE,
        min_val=0.0, max_val=105.0,
        rtd_type=nidaqmx.constants.RTDType.CUSTOM,
        current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL, current_excit_val=1e-3,
        r_0=100)
    rtd_channel.ni.ai_rtd_a = 3.9083e-3 # This is how the original rtd was defined for the calibration
    rtd_channel.ni.ai_rtd_b = -577.5e-9
    rtd_channel.ni.ai_rtd_c = -4.183e-12

    current_channel.add_to_task(
        task=experiment, channel_specification='current_chan',
        terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
        min_val=0, max_val=50,
        shunt_resistor_loc=nidaqmx.constants.CurrentShuntResistorLocation.EXTERNAL,
        ext_shunt_resistor_val=4e-3
    )
    led_reading_channel.add_to_task(
        task=experiment, channel_specification='voltage_chan',
        terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
        min_val=0.0, max_val=7.0
    )
    thermocouple_channel.add_to_task(
        task=experiment, channel_specification='thrmcpl_chan',
        thermocouple_type=nidaqmx.constants.ThermocoupleType.K,
        min_val=10.0, max_val=1000.0,
        cjc_val=23.0
    )

    experiment.timing.cfg_samp_clk_timing(sample_rate, sample_mode=sample_mode)
    experiment.start()

    print_if_must(('anything', 'info'), f'experiment samp_clk_rate: {experiment.timing.samp_clk_rate}')

#%%
    # -------------------------------------------------------
    # Run experiment
    # -------------------------------------------------------
    first = True
    should_continue = True
    iter_count = 0
    readings = {}
    start = time.time()
    while first or should_continue:
        # -------------------------------------------------------
        # Header
        # -------------------------------------------------------
        print_if_must(('anything', 'info'), f'> Iteration {iter_count}')
#%%
        # -------------------------------------------------------
        # Time measurement
        # -------------------------------------------------------
        elapsed_time = np.array([time.time() - start])
#%%
        # -------------------------------------------------------
        # Read data
        # -------------------------------------------------------
        readings[experiment.name] = experiment.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)

#%%
        # -------------------------------------------------------
        # Process data
        # -------------------------------------------------------
        # Electric data:
        voltage_readings = {key: chan.read(experiment, readings, dtype=np.array) for key, chan in voltage_channels.items()}
        voltage_readings_values = np.array(list(voltage_readings.values()))
        voltage = np.sum(voltage_readings_values, axis=0) if voltage_readings_values.size > 0 else voltage_readings_values
        current = current_channel.read(experiment, readings, dtype=np.array)
        power = voltage * current
        flux = power / wire_surface_area
        # resistance = calculate_resistance(voltage, current)

        # Thermal data:
        rtd_read_value = rtd_channel.read(experiment, readings, dtype=np.array)
        rtd_temperature = rtd_read_value
        if rtd_temperature.size > 0:
            rtd_temperature = calibrated_polynomial(rtd_temperature)

        wire_temperature = thermocouple_channel.read(experiment, readings, dtype=np.array)
        # wire_temperature_corrected = wire_temperature + wire_temperature_correction
        # wire_temperature_from_resistance = calculate_temperature(resistance)

        superheat = wire_temperature - rtd_temperature

        # LED data:
        led_voltage = led_reading_channel.read(experiment, readings, dtype=np.array)

#%%
        # -------------------------------------------------------
        # Saving
        # -------------------------------------------------------
        local_data = {
            'Voltage [V]': voltage,
            'Current [A]': current,
            'Power [W]': power,
            'Flux [W/m^2]': flux,
            'Flux [W/cm^2]': flux/100**2,
            # 'Resistance [Ohm]': resistance,
            'Bulk Temperature [deg C]': rtd_temperature,
            'LED Voltage [V]': led_voltage,
            'Wire Temperature [deg C]': wire_temperature,
            'Superheat [deg C]': superheat,
            # 'Temperature from Resistance [deg C]': wire_temperature_from_resistance,
            # 'Wire Temperature (corrected) [deg C]': wire_temperature_corrected,
        }
        n_values = min([local.size for local in local_data.values()])

        # Time measurement
        if n_values > 1:
            now = time.time() - start
            sampled_period = (now - elapsed_time) / n_values
            elapsed_time = np.pad(elapsed_time, (0, n_values - elapsed_time.size), 'edge') + np.arange(n_values)*sampled_period
        print_if_must(('anything', 'info'), f'n_values: {n_values}')
        print_if_must(('anything', 'elapsed time'), f'Elapsed time [s]: {elapsed_time}')

        local_data = {
            **{
                'Time instant': np.array([datetime.fromtimestamp(start + et) for et in elapsed_time]),
                'Elapsed time': elapsed_time,
            },
            **local_data
        }

        # TODO: here
        if measure_loop_time:
            if first:
                loop_time = np.zeros(n_values)
            else:
                if elapsed_time.size > 1:
                    print(previous_elapsed_time)
                    print(elapsed_time)
                    print(previous_elapsed_time[-1].shape)
                    print(elapsed_time[:-1].shape)
                    loop_time = elapsed_time - np.c_[previous_elapsed_time[-1], elapsed_time[:-1]]
                else:
                    loop_time = elapsed_time - previous_elapsed_time[-1]
                print_if_must(('anything', 'elapsed time'), f'Loop time: {loop_time}')
            previous_elapsed_time = elapsed_time
            local_data['Loop time'] = loop_time

        # Data
        keys = local_data.keys()
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

#%%
        # -------------------------------------------------------
        # Writing to file
        # -------------------------------------------------------
        if save:
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
        print_if_must(('anything',), '>>')
        print_if_must(('anything', 'voltage'), f'>> Voltage [V]: {voltage}', conds=[voltage.size > 0])
        print_if_must(('anything', 'current'), f'>> Current [A]: {current}', conds=[current.size > 0])
        print_if_must(('anything', 'power'), f'>> Power [W]: {power}', conds=[power.size > 0])
        print_if_must(('anything', 'flux'), f'>> Flux [W/m^2]: {flux}', conds=[flux.size > 0])
        # print_if_must(('anything', 'resistance'), f'>> Resistance [Ohm]: {resistance}', conds=[resistance.size > 0])
        print_if_must(('anything', 'bulk temperature'), f'>> Bulk Temperature [°C]: {rtd_temperature}', conds=[rtd_temperature.size > 0])
        print_if_must(('anything', 'led read state'), f'>> LED: {led_voltage}', conds=[led_voltage.size > 0])
        print_if_must(('anything', 'wire temperature'), f'>> Wire Temperature [°C]: {wire_temperature}', conds=[wire_temperature.size > 0])
        # print_if_must(('anything', 'temperature from resistance'), f'>> Temperature from Resistance [deg C]: {wire_temperature_from_resistance}', conds=[wire_temperature_from_resistance.size > 0])
        # print_if_must(('anything', 'wire temperature corrected'), f'>> Wire Temperature (corrected) [deg C]: {wire_temperature_corrected}', conds=[wire_temperature_corrected.size > 0])

#%%
        # -------------------------------------------------------
        # Plotting
        # -------------------------------------------------------
        if should_plot:
            number_of_variables = len(local_data)

            if first:
                win = pg.GraphicsWindow(title=filepath.name)
                ps = {
                    'Bulk Temperature [deg C]': win.addPlot(title='Bulk Temperature', row=0, col=0, rowspan=2, colspan=1),
                    'Wire Temperature [deg C]': win.addPlot(title='Wire Temperature', row=0, col=1, rowspan=2, colspan=1),
                    'Superheat [deg C]': win.addPlot(title='Wire Temperature', row=0, col=2, rowspan=2, colspan=1),
                    'Power [W]': win.addPlot(title='Power', row=3, col=2, colspan=1),
                    # 'Flux [W/m^2]': win.addPlot(title='Flux [W/m^2]', row=1, col=2, colspan=1),
                    'Flux [W/cm^2]': win.addPlot(title='Flux [W/cm^2]', row=0, col=3, rowspan=2, colspan=3),
                    'Voltage [V]': win.addPlot(title='Voltage', row=3, col=0),
                    'Current [A]': win.addPlot(title='Current', row=3, col=1),
                    # 'Resistance [Ohm]': win.addPlot(title='Resistance', row=2, col=4),
                    'LED Voltage [V]': win.addPlot(title='LED Voltage', row=3, col=4),
                    # 'Wire Temperature (corrected) [deg C]': win.addPlot(title='Wire Temperature (corrected) [deg C]', row=2, col=2, rowspan=2, colspan=2),
                    # 'Temperature from Resistance [deg C]': win.addPlot(title='Temperature from Resistance [deg C]', row=2, col=6),
                }
                curves = {key: ps[key].plot() for key in ps}

                x = {key: np.zeros(x_axis_size) for key in ps}
                y = {key: np.zeros(x_axis_size) for key in ps}

                for key in ps:
                    ps[key].setLabel('bottom', 'Iteration')
                    ps[key].setLabel('left', key)

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
                ps[key].setTitle(f'{key} [{y[key][-1]}]')

            QtGui.QApplication.processEvents()

#%%
        # -------------------------------------------------------
        # Finish iteration
        # -------------------------------------------------------
        iter_count += 1
        first = False
        if not read_continuously:
            should_continue = iter_count <= maximum_number_of_iterations


if should_plot:
    ### END QtApp ####
    pg.QtGui.QApplication.exec_() # you MUST put this at the end
    ##################



#%%
# -------------------------------------------------------
# Plot Results
# -------------------------------------------------------
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# datatype = [
#     ('index', numpy.float32),
#     ('floati', numpy.float32),
#     ('floatq', numpy.float32)
# ]
datatype = [
    ('Time instant', numpy.float32),
    ('Elapsed time', numpy.float32),
    ('Voltage', numpy.float32),
    ('Current', numpy.float32),
    ('Power', numpy.float32),
    ('Flux', numpy.float32),
    ('Resistance', numpy.float32),
    ('Bulk Temperature', numpy.float32),
    ('LED Voltage', numpy.float32),
    ('Wire Temperature', numpy.float32),
]
# datatype = numpy.float32

filename = Path() / 'Experiment Output 20-01-2020' / 'Experiment 0 -- 17-56.csv'

# data = numpy.memmap(filename, datatype, 'r')
# plt.plot(data['Elapsed time'], data['floatq'], 'r,')
# plt.grid(True)
# plt.title("Signal-Diagram")
# plt.xlabel("Sample")
# plt.ylabel("In-Phase")
# plt.savefig('foo.png')