import csv
import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import modin.pandas as pd
import nidaqmx
import numpy as np
import scipy.interpolate

from boiling_learning.daq import Channel, ChannelType, Device
from boiling_learning.utils.geometry import Cylinder
from boiling_learning.utils.units import unit_registry as u
from boiling_learning.utils.utils import (
    PathLike,
    ensure_dir,
    ensure_parent,
    print_verbose,
)

here = Path().resolve()
DEFAULT_EXPERIMENTS_DIR = here / 'experiments'
DEFAULT_CALIBRATION_FILEPATH = (
    here.parent
    / 'resources'
    / 'experimental-set-calibration'
    / 'Processing'
    / 'coefficients.csv'
)


def main(
    experiments_dir: PathLike = DEFAULT_EXPERIMENTS_DIR,
    calibration_filepath: PathLike = DEFAULT_CALIBRATION_FILEPATH,
) -> None:
    import pyqtgraph as pg
    from pyqtgraph import GraphicsWindow
    from pyqtgraph.Qt import QtGui

    output_dir_pattern = str(
        Path(
            experiments_dir, 'Experiment %Y-%m-%d %H-%M{optional_index}'
        ).resolve()
    )

    calibration_filepath = Path(calibration_filepath)

    # -------------------------------------------------------
    # Settings
    # -------------------------------------------------------
    # ribbon or wire?
    sample = Cylinder(length=6.5 * u.cm, diameter=0.518 * u.mm)

    read_continuously = True
    maximum_number_of_iterations = 21

    print('Cross section area:', sample.cross_section_area.to_base_units())
    print('Surface area:', sample.surface_area.to_base_units())

    must_print = {
        'anything': True,  # Use false to disallow printing at all
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
        'temperature from resistance': False,
    }

    sample_rate = 10  # Hz
    sample_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS
    sample_period = 1 / sample_rate  # s
    sleeping_time = 0.5 * sample_period

    write_period = 100

    def optional_index_format(counter):
        return f' ({counter})' if counter != 0 else ''

    filename_pattern = r'data.csv'

    x_axis_size = 10 * sample_rate

    should_plot = True

    measure_loop_time = False
    save = False

    # Pairs (T, f) where T is the measured temperature in °C and f is the factor to multiply the wire resistance
    factor_table = pd.DataFrame(
        {
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
            ],
        }
    )

    f_to_T = scipy.interpolate.interp1d(
        factor_table['Factor'],
        factor_table['Temperature'],
        copy=False,
        fill_value='extrapolate',
    )
    T_to_f = scipy.interpolate.interp1d(
        factor_table['Temperature'],
        factor_table['Factor'],
        copy=False,
        fill_value='extrapolate',
    )

    u.Quantity(20, u.degC)
    reference_resistivity = 650 * u.ohm * u.cmil / u.foot
    reference_resistance = (
        reference_resistivity * sample.length / sample.cross_section_area
    )

    def calculate_resistance(voltage, current):
        return np.where(
            np.abs(current) >= 1e-8,
            np.abs(voltage / current),
            reference_resistance,
        )

    def calculate_temperature(resistance):
        factor = resistance / reference_resistance
        return f_to_T(factor)

    @functools.lru_cache(maxsize=128)
    def correct_wire_temperature(reference_file):
        df = pd.read_csv(
            reference_file,
            usecols=['Bulk Temperature [deg C]', 'Wire Temperature [deg C]'],
        )
        mean_bulk_temperature = df['Bulk Temperature [deg C]'].mean()
        mean_wire_temperature = df['Wire Temperature [deg C]'].mean()

        return mean_bulk_temperature - mean_wire_temperature

    # wire_temperature_correction_reference_file = Path() / 'experiments' / 'Experiment Output 2020-02-14' / 'Experiment 10-03 (0).csv'
    # wire_temperature_correction_reference_file = Path() / 'experiments' / 'Experiment Output 2020-02-18' / 'Experiment 16-44 (0).csv'
    # wire_temperature_correction = correct_wire_temperature(wire_temperature_correction_reference_file)

    # For timestamps: <https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z000000kJy2SAE&l=pt-BR>

    """
    Support definitions -------------------------------------------------------
    """

    def format_output_dir(output_dir_pattern):
        full_dir_pattern = str(
            Path('experiments', datetime.now().strftime(output_dir_pattern))
        )

        def format_time(dirpattern, counter):
            return datetime.now().strftime(
                dirpattern.format(
                    index=counter,
                    optional_index=optional_index_format(counter),
                )
            )

        def substituted_output_dir(dirpattern, counter):
            return Path(format_time(full_dir_pattern, counter))

        counter = 0
        if any(
            key in output_dir_pattern
            for key in ('{index}', '{optional_index}')
        ):
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

    def print_if_must(
        keys, *args, conds: Iterable[bool] = (), **kwargs
    ) -> None:
        cond = all(conds) and all(must_print.get(key, False) for key in keys)
        print_verbose(cond, *args, **kwargs)

    # -------------------------------------------------------
    # Channel definitions
    # -------------------------------------------------------
    voltage_channels = {
        'Orange Resistor': Channel(
            Device('cDAQ1Mod4'),
            'ai3',
            'Orange Resistor',
            ChannelType.ANALOG,
            ChannelType.INPUT,
        ),
        'Blue Resistor': Channel(
            Device('cDAQ1Mod4'),
            'ai4',
            'Blue Resistor',
            ChannelType.ANALOG,
            ChannelType.INPUT,
        ),
        'Yellow Resistor': Channel(
            Device('cDAQ1Mod4'),
            'ai5',
            'Yellow Resistor',
            ChannelType.ANALOG,
            ChannelType.INPUT,
        ),
    }
    current_channel = Channel(
        Device('cDAQ1Mod4'),
        'ai0',
        'Current Reading Channel',
        ChannelType.ANALOG,
        ChannelType.INPUT,
    )
    rtd_channel = Channel(
        Device('cDAQ1Mod6'),
        'ai0',
        'RTD Reading Channel',
        ChannelType.ANALOG,
        ChannelType.INPUT,
    )
    led_reading_channel = Channel(
        Device('cDAQ1Mod4'),
        'ai7',
        'LED Reading Channel',
        ChannelType.ANALOG,
        ChannelType.INPUT,
    )
    thermocouple_channel = Channel(
        Device('cDAQ1Mod1'),
        'ai1',
        'Thermocouple Reading Channel',
        ChannelType.ANALOG,
        ChannelType.INPUT,
    )

    """
    Print system information -------------------------------------------------------
    """
    system = nidaqmx.system.System.local()
    print_if_must(('anything', 'info'), f'> {system.driver_version}')
    print_if_must(
        ('anything', 'info'),
        f'> Available devices in system.devices: {tuple(system.devices)}',
    )
    print_if_must(
        ('anything', 'info'),
        f'> AI channels in system.devices: { {x: [y.name for y in x.ai_physical_chans] for x in system.devices} }',
    )
    print_if_must(
        ('anything', 'info'),
        f'> DO ports in system.devices: { {x: [y.name for y in x.do_ports] for x in system.devices} }',
    )
    print_if_must(
        ('anything', 'info'), f'> Types in ChannelType: {tuple(ChannelType)}'
    )

    """
    Load calibration polynomial -------------------------------------------------------
    """
    coefficients = calibration_filepath.read_text().splitlines()
    coefficients = [float(x.strip()) for x in coefficients]
    calibrated_polynomial = np.poly1d(list(reversed(coefficients)))
    print_if_must(
        ('anything', 'info'),
        f'> Calibrated polynomial:\n{calibrated_polynomial}',
    )

    # -------------------------------------------------------
    # Initialize
    # -------------------------------------------------------
    output_dir = ensure_dir(format_output_dir(output_dir_pattern))
    filepath = output_dir / format_filename(output_dir, filename_pattern)
    filepath = ensure_parent(filepath)
    print_if_must(('anything', 'info'), f'> File path: {filepath}')

    if should_plot:
        ### START QtApp #####
        QtGui.QApplication([])
        # see <https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph>
        ####################

    with filepath.open('w', newline='') as output_file, nidaqmx.Task(
        'Experiment'
    ) as experiment:
        output_writer = csv.writer(output_file)

        # -------------------------------------------------------
        # Setup channels
        # -------------------------------------------------------
        for (
            channel_nickname,
            voltage_reading_channel,
        ) in voltage_channels.items():
            voltage_channels[channel_nickname].add_to_task(
                task=experiment,
                channel_specification='voltage_chan',
                terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
                min_val=0.0,
                max_val=10.0,
            )

        rtd_channel.add_to_task(
            task=experiment,
            channel_specification='rtd_chan',
            resistance_config=nidaqmx.constants.ResistanceConfiguration.FOUR_WIRE,
            min_val=0.0,
            max_val=105.0,
            rtd_type=nidaqmx.constants.RTDType.CUSTOM,
            current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
            current_excit_val=1e-3,
            r_0=100,
        )
        rtd_channel.ni.ai_rtd_a = 3.9083e-3  # This is how the original rtd was defined for the calibration
        rtd_channel.ni.ai_rtd_b = -577.5e-9
        rtd_channel.ni.ai_rtd_c = -4.183e-12

        current_channel.add_to_task(
            task=experiment,
            channel_specification='current_chan',
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=0,
            max_val=50,
            shunt_resistor_loc=nidaqmx.constants.CurrentShuntResistorLocation.EXTERNAL,
            ext_shunt_resistor_val=4e-3,
        )
        led_reading_channel.add_to_task(
            task=experiment,
            channel_specification='voltage_chan',
            terminal_config=nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL,
            min_val=0.0,
            max_val=7.0,
        )
        thermocouple_channel.add_to_task(
            task=experiment,
            channel_specification='thrmcpl_chan',
            thermocouple_type=nidaqmx.constants.ThermocoupleType.K,
            min_val=10.0,
            max_val=1000.0,
            cjc_val=23.0,
        )

        experiment.timing.cfg_samp_clk_timing(
            sample_rate, sample_mode=sample_mode
        )
        experiment.start()

        print_if_must(
            ('anything', 'info'),
            f'experiment samp_clk_rate: {experiment.timing.samp_clk_rate}',
        )

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
            # -------------------------------------------------------
            # Time measurement
            # -------------------------------------------------------
            elapsed_time = np.array([time.time() - start])
            # -------------------------------------------------------
            # Read data
            # -------------------------------------------------------
            readings[experiment.name] = experiment.read(
                number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE
            )

            # -------------------------------------------------------
            # Process data
            # -------------------------------------------------------
            # Electric data:
            voltage_readings = {
                key: chan.read(experiment, readings, dtype=np.array)
                for key, chan in voltage_channels.items()
            }
            voltage_readings_values = np.array(list(voltage_readings.values()))
            voltage = (
                np.sum(voltage_readings_values, axis=0)
                if voltage_readings_values.size > 0
                else voltage_readings_values
            ) * u.V
            current = (
                current_channel.read(experiment, readings, dtype=np.array)
                * u.A
            )
            power = voltage * current
            flux = power / sample.surface_area
            # resistance = calculate_resistance(voltage, current)

            # Thermal data:
            rtd_read_value = rtd_channel.read(
                experiment, readings, dtype=np.array
            )
            rtd_temperature = rtd_read_value
            if rtd_temperature.size > 0:
                rtd_temperature = calibrated_polynomial(rtd_temperature)

            wire_temperature = thermocouple_channel.read(
                experiment, readings, dtype=np.array
            )
            # wire_temperature_corrected = wire_temperature + wire_temperature_correction
            # wire_temperature_from_resistance = calculate_temperature(resistance)

            superheat = wire_temperature - rtd_temperature

            # LED data:
            led_voltage = led_reading_channel.read(
                experiment, readings, dtype=np.array
            )

            # -------------------------------------------------------
            # Saving
            # -------------------------------------------------------
            local_data = {
                'Voltage [V]': voltage.m_as(u.V),
                'Current [A]': current.m_as(u.A),
                'Power [W]': power.m_as(u.W),
                'Flux [W/m^2]': flux.m_as(u.W / u.m ** 2),
                'Flux [W/cm^2]': flux.m_as(u.W / u.cm ** 2),
                # 'Resistance [Ohm]': resistance,
                'Bulk Temperature [deg C]': rtd_temperature,
                'LED Voltage [V]': led_voltage,
                'Wire Temperature [deg C]': wire_temperature,
                'Superheat [deg C]': superheat,
                # 'Temperature from Resistance [deg C]': wire_temperature_from_resistance,
                # 'Wire Temperature (corrected) [deg C]': wire_temperature_corrected,
            }
            n_values = min(local.size for local in local_data.values())

            # Time measurement
            if n_values > 1:
                now = time.time() - start
                sampled_period = (now - elapsed_time) / n_values
                elapsed_time = (
                    np.pad(
                        elapsed_time, (0, n_values - elapsed_time.size), 'edge'
                    )
                    + np.arange(n_values) * sampled_period
                )
            print_if_must(('anything', 'info'), f'n_values: {n_values}')
            print_if_must(
                ('anything', 'elapsed time'),
                f'Elapsed time [s]: {elapsed_time}',
            )

            local_data = {
                **{
                    'Time instant': np.array(
                        [
                            datetime.fromtimestamp(start + et)
                            for et in elapsed_time
                        ]
                    ),
                    'Elapsed time': elapsed_time,
                },
                **local_data,
            }

            # TODO: here
            if measure_loop_time:
                previous_elapsed_time = None
                if first:
                    loop_time = np.zeros(n_values)
                else:
                    if elapsed_time.size > 1:
                        print(previous_elapsed_time)
                        print(elapsed_time)
                        print(previous_elapsed_time[-1].shape)
                        print(elapsed_time[:-1].shape)
                        loop_time = (
                            elapsed_time
                            - np.c_[
                                previous_elapsed_time[-1], elapsed_time[:-1]
                            ]
                        )
                    else:
                        loop_time = elapsed_time - previous_elapsed_time[-1]
                    print_if_must(
                        ('anything', 'elapsed time'), f'Loop time: {loop_time}'
                    )
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
                    print_if_must(
                        ('anything', 'sleeping'), f'{key} is causing sleep'
                    )
                    continue_key = True
            if continue_key:
                time.sleep(sleeping_time)
                continue

            # -------------------------------------------------------
            # Writing to file
            # -------------------------------------------------------
            if save:
                if first:
                    output_writer.writerow(keys)

                if iter_count % write_period == 0:
                    print_if_must(
                        ('anything', 'writing'),
                        '>> Writing to file...',
                        end='',
                    )

                    output_writer.writerows(zip(*[data[key] for key in keys]))
                    data = generate_empty_copy(local_data)

                    print_if_must(('anything', 'writing'), '>> Done')

            """
            Printing -------------------------------------------------------
            """
            print_if_must(('anything',), '>>')
            print_if_must(
                ('anything', 'voltage'),
                f'>> Voltage [V]: {voltage}',
                conds=[voltage.size > 0],
            )
            print_if_must(
                ('anything', 'current'),
                f'>> Current [A]: {current}',
                conds=[current.size > 0],
            )
            print_if_must(
                ('anything', 'power'),
                f'>> Power [W]: {power}',
                conds=[power.size > 0],
            )
            print_if_must(
                ('anything', 'flux'),
                f'>> Flux [W/m^2]: {flux}',
                conds=[flux.size > 0],
            )
            # print_if_must(('anything', 'resistance'), f'>> Resistance [Ohm]: {resistance}', conds=[resistance.size > 0])
            print_if_must(
                ('anything', 'bulk temperature'),
                f'>> Bulk Temperature [°C]: {rtd_temperature}',
                conds=[rtd_temperature.size > 0],
            )
            print_if_must(
                ('anything', 'led read state'),
                f'>> LED: {led_voltage}',
                conds=[led_voltage.size > 0],
            )
            print_if_must(
                ('anything', 'wire temperature'),
                f'>> Wire Temperature [°C]: {wire_temperature}',
                conds=[wire_temperature.size > 0],
            )
            # print_if_must(('anything', 'temperature from resistance'), f'>> Temperature from Resistance [deg C]: {wire_temperature_from_resistance}', conds=[wire_temperature_from_resistance.size > 0])
            # print_if_must(('anything', 'wire temperature corrected'), f'>> Wire Temperature (corrected) [deg C]: {wire_temperature_corrected}', conds=[wire_temperature_corrected.size > 0])

            # -------------------------------------------------------
            # Plotting
            # -------------------------------------------------------
            if should_plot:
                if first:
                    win = GraphicsWindow(title=filepath.name)
                    ps = {
                        'Bulk Temperature [deg C]': win.addPlot(
                            title='Bulk Temperature',
                            row=0,
                            col=0,
                            rowspan=2,
                            colspan=1,
                        ),
                        'Wire Temperature [deg C]': win.addPlot(
                            title='Wire Temperature',
                            row=0,
                            col=1,
                            rowspan=2,
                            colspan=1,
                        ),
                        'Superheat [deg C]': win.addPlot(
                            title='Wire Temperature',
                            row=0,
                            col=2,
                            rowspan=2,
                            colspan=1,
                        ),
                        'Power [W]': win.addPlot(
                            title='Power', row=3, col=2, colspan=1
                        ),
                        # 'Flux [W/m^2]': win.addPlot(title='Flux [W/m^2]', row=1, col=2, colspan=1),
                        'Flux [W/cm^2]': win.addPlot(
                            title='Flux [W/cm^2]',
                            row=0,
                            col=3,
                            rowspan=2,
                            colspan=3,
                        ),
                        'Voltage [V]': win.addPlot(
                            title='Voltage', row=3, col=0
                        ),
                        'Current [A]': win.addPlot(
                            title='Current', row=3, col=1
                        ),
                        # 'Resistance [Ohm]': win.addPlot(title='Resistance', row=2, col=4),
                        'LED Voltage [V]': win.addPlot(
                            title='LED Voltage', row=3, col=4
                        ),
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
                    x[key][:n_values] = elapsed_time
                    x[key] = np.roll(x[key], -n_values)

                    y[key][:n_values] = local_data[key]
                    y[key] = np.roll(y[key], -n_values)

                    if iter_count < x_axis_size:
                        curves[key].setData(
                            x[key][x_axis_size - iter_count + 1 :],
                            y[key][x_axis_size - iter_count + 1 :],
                        )
                        curves[key].setPos(0, 0)
                    else:
                        curves[key].setData(x[key], y[key])
                        curves[key].setPos(iter_count - x_axis_size, 0)
                    ps[key].setTitle(f'{key} [{y[key][-1]}]')

                QtGui.QApplication.processEvents()

            # -------------------------------------------------------
            # Finish iteration
            # -------------------------------------------------------
            iter_count += 1
            first = False
            if not read_continuously:
                should_continue = iter_count <= maximum_number_of_iterations

    if should_plot:
        ### END QtApp ####
        pg.QtGui.QApplication.exec_()  # you MUST put this at the end
        ##################

    # -------------------------------------------------------
    # Plot Results
    # -------------------------------------------------------
    mpl.use('Agg')

    # datatype = [
    #     ('index', np.float32),
    #     ('floati', np.float32),
    #     ('floatq', np.float32)
    # ]
    datatype = [
        ('Time instant', np.float32),
        ('Elapsed time', np.float32),
        ('Voltage', np.float32),
        ('Current', np.float32),
        ('Power', np.float32),
        ('Flux', np.float32),
        ('Resistance', np.float32),
        ('Bulk Temperature', np.float32),
        ('LED Voltage', np.float32),
        ('Wire Temperature', np.float32),
    ]
    # datatype = np.float32

    # filename = Path('Experiment Output 20-01-2020', 'Experiment 0 -- 17-56.csv')
    # data = np.memmap(filename, datatype, 'r')
    # plt.plot(data['Elapsed time'], data['floatq'], 'r,')
    # plt.grid(True)
    # plt.title("Signal-Diagram")
    # plt.xlabel("Sample")
    # plt.ylabel("In-Phase")
    # plt.savefig('foo.png')


if __name__ == '__main__':
    main()
