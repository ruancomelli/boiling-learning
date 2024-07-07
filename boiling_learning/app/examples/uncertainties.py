from pathlib import Path

import numpy as np
import pandas as pd
import typer
from pint import Quantity
from rich.console import Console
from rich.panel import Panel

from boiling_learning.app.datasets.raw.boiling1d import boiling_data_path
from boiling_learning.utils.geometry import Cylinder, RectangularPrism
from boiling_learning.utils.units import unit_registry as u

app = typer.Typer()
console = Console()

MICROMETER_UNCERTAINTY = 5 * u.micrometer
CALIPER_UNCERTAINTY = 0.02 * u.millimeter

LARGE_WIRE_DIAMETER_MEASUREMENTS = [
    (value * u.micrometer).plus_minus(MICROMETER_UNCERTAINTY)
    for value in (510, 510, 509, 510, 510)
]

SMALL_WIRE_DIAMETER_MEASUREMENTS = [
    (value * u.micrometer).plus_minus(MICROMETER_UNCERTAINTY)
    for value in (251, 251, 250, 250, 250)
]

RIBBON_THICKNESS_MEASUREMENTS = [
    (value * u.micrometer).plus_minus(MICROMETER_UNCERTAINTY) for value in (79, 80, 79, 79, 79)
]

RIBBON_WIDTH_MEASUREMENTS = [
    (value * u.micrometer).plus_minus(MICROMETER_UNCERTAINTY)
    for value in (1590, 1588, 1590, 1591, 1590)
]

LENGTH_MEASUREMENTS = [
    (6.502 * u.centimeter).plus_minus(CALIPER_UNCERTAINTY),
    (6.503 * u.centimeter).plus_minus(CALIPER_UNCERTAINTY),
    (6.503 * u.centimeter).plus_minus(CALIPER_UNCERTAINTY),
]

# From https://www.artisantg.com/info/NI_9205_Manual.pdf
VOLTAGE_MODULE_UNCERTAINTY_10V = 6230 * u.microvolt
VOLTAGE_MODULE_UNCERTAINTY_02V = 174 * u.microvolt

VOLTAGE_COLUMN = 'Voltage [V]'
CURRENT_COLUMN = 'Current [A]'
POWER_COLUMN = 'Power [W]'
FLUX_COLUMN = 'Flux [W/cm**2]'
NOMINAL_POWER_COLUMN = 'nominal_power'

HEAT_FLUX_UNIT = u.watt / u.centimeter**2
COVERAGE_FACTOR = 2


@app.command()
def main() -> None:
    case_name_mapping = {
        'LW': 'case 1',
        'SW': 'case 2',
        'HR': 'case 3',
        'VR': 'case 4',
    }

    max_currents = {
        'LW': 18 * u.ampere,
        'SW': 7 * u.ampere,
        'HR': 18 * u.ampere,
        'VR': 17 * u.ampere,
    }

    case_data = {
        case_name: _filter_outliers(
            _load_data_from_csvs(boiling_data_path() / case_directory_name / 'dataframes')
        )
        for case_name, case_directory_name in case_name_mapping.items()
    }

    console.print(Panel('Starting uncertainties example'))

    sample_length = np.array(LENGTH_MEASUREMENTS).mean()
    large_wire_diameter = np.array(LARGE_WIRE_DIAMETER_MEASUREMENTS).mean()
    small_wire_diameter = np.array(SMALL_WIRE_DIAMETER_MEASUREMENTS).mean()
    ribbon_thickness = np.array(RIBBON_THICKNESS_MEASUREMENTS).mean()
    ribbon_width = np.array(RIBBON_WIDTH_MEASUREMENTS).mean()

    console.print(
        'Sample length:',
        sample_length,
        f'(uncertainty: {sample_length.error * COVERAGE_FACTOR})',
    )
    console.print(
        'Large wire diameter:',
        large_wire_diameter,
        f'(uncertainty: {large_wire_diameter.error * COVERAGE_FACTOR})',
    )
    console.print(
        'Small wire diameter:',
        small_wire_diameter,
        f'(uncertainty: {small_wire_diameter.error * COVERAGE_FACTOR})',
    )
    console.print(
        'Ribbon thickness:',
        ribbon_thickness,
        f'(uncertainty: {ribbon_thickness.error * COVERAGE_FACTOR})',
    )
    console.print(
        'Ribbon width:',
        ribbon_width,
        f'(uncertainty: {ribbon_width.error * COVERAGE_FACTOR})',
    )

    large_wire = Cylinder(sample_length, large_wire_diameter)
    small_wire = Cylinder(sample_length, small_wire_diameter)
    ribbon = RectangularPrism(sample_length, ribbon_width, ribbon_thickness)

    large_wire_surface_area = large_wire.surface_area()
    small_wire_surface_area = small_wire.surface_area()
    ribbon_surface_area = ribbon.surface_area()

    large_wire_surface_area_uncertainty = large_wire_surface_area.error * COVERAGE_FACTOR
    console.print(
        'Large wire surface area:',
        large_wire_surface_area.to(u.cm**2),
        f'(uncertainty: {large_wire_surface_area_uncertainty})',
        f'(relative uncertainty: {(large_wire_surface_area_uncertainty / large_wire_surface_area).magnitude:.2%})',
    )

    small_wire_surface_area_uncertainty = small_wire_surface_area.error * COVERAGE_FACTOR
    console.print(
        'Small wire surface area:',
        small_wire_surface_area.to(u.cm**2),
        f'(uncertainty: {small_wire_surface_area.error * COVERAGE_FACTOR})',
        f'(relative uncertainty: {(small_wire_surface_area_uncertainty / small_wire_surface_area).magnitude:.2%})',
    )

    ribbon_surface_area_uncertainty = ribbon_surface_area.error * COVERAGE_FACTOR
    console.print(
        'Ribbon surface area:',
        ribbon_surface_area.to(u.cm**2),
        f'(uncertainty: {ribbon_surface_area.error * COVERAGE_FACTOR})',
        f'(relative uncertainty: {(ribbon_surface_area_uncertainty / ribbon_surface_area).magnitude:.2%})',
    )

    case_surfaces = {
        'LW': large_wire,
        'SW': small_wire,
        'HR': ribbon,
        'VR': ribbon,
    }

    shunt_resistance = (4 * u.milliohm).plus_minus(0.5 / 100, relative=True)

    shunt_current_samples = {
        case_name: data[CURRENT_COLUMN].values.clip(0, max_currents[case_name].m_as(u.ampere))
        * u.ampere
        for case_name, data in case_data.items()
    }

    shunt_voltages = {
        case_name: _array_with_uncertainty(
            (shunt_resistance.value * current).to(u.millivolt), VOLTAGE_MODULE_UNCERTAINTY_02V
        )
        for case_name, current in shunt_current_samples.items()
    }

    console.print('Shunt resistance:', shunt_resistance)

    currents = {
        case_name: np.array([current.to(u.ampere) for current in voltage / shunt_resistance])
        for case_name, voltage in shunt_voltages.items()
    }

    voltages = {
        case_name: _array_with_uncertainty(
            data[VOLTAGE_COLUMN].values * u.volt, VOLTAGE_MODULE_UNCERTAINTY_10V
        )
        for case_name, data in case_data.items()
    }

    thermal_powers = {
        case_name: np.array([power.to(u.watt) for power in voltage * current])
        for case_name, (voltage, current) in zip(
            case_data, zip(voltages.values(), currents.values())
        )
    }

    heat_fluxes = {
        case_name: np.array(
            [flux.to(u.watt / u.cm**2) for flux in thermal_power / surface.surface_area()]
        )
        for case_name, (thermal_power, surface) in zip(
            case_data, zip(thermal_powers.values(), case_surfaces.values())
        )
    }

    for case_name, fluxes in heat_fluxes.items():
        standard_uncertainty = max(flux.error for flux in fluxes)
        console.print(
            f"Maximum standard uncertainty for '{case_name}':",
            standard_uncertainty,
        )
        console.print(
            f"Maximum expanded uncertainty for '{case_name}':",
            COVERAGE_FACTOR * standard_uncertainty,
        )


def _array_with_uncertainty(
    array: np.ndarray, uncertainty: Quantity | float, relative: bool = False
) -> np.ndarray:
    return np.array([value.plus_minus(uncertainty, relative=relative) for value in array])


def _filter_outliers(data: pd.DataFrame) -> pd.DataFrame:
    return data[(data['Power [W]'] - data['nominal_power']).abs() < 5]


def _load_data_from_csvs(path: Path) -> pd.DataFrame:
    return pd.concat(
        pd.read_csv(
            csv_file_path,
            usecols=[
                VOLTAGE_COLUMN,
                CURRENT_COLUMN,
                POWER_COLUMN,
                FLUX_COLUMN,
                NOMINAL_POWER_COLUMN,
            ],
        )
        for csv_file_path in path.glob('*.csv')
    )
