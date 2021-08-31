import csv
import pathlib
import re

import numpy as np
import pint
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# Support functions
def regularized_untangled_file_name(details):
    details['day'] = int(details['day'])
    details['month'] = int(details['month'])
    details['year'] = int(details['year'])
    details['hour'] = int(details['hour'])
    details['minute'] = int(details['minute'])
    details['temperature_unity'] = float(details['temperature_unity'])
    details['temperature_decy'] = float(details['temperature_decy'])
    details['reference_temperature'] = (
        details['temperature_unity'] + 0.01 * details['temperature_decy']
    )

    return details


def untangle_file_name(file_name):
    pattern = re.compile(
        r'RTD Temperature (?P<day>[0-9]{2})-(?P<month>[0-9]{2})-(?P<year>[0-9]{4}) -- (?P<hour>[0-9]{2})(?P<minute>[0-9]{2}) -- (?P<temperature_unity>[0-9]{1,2})_(?P<temperature_decy>[0-9]{2})C -- (?P<attempt>[0-9]+)\.(?P<extension>.*)'
    )
    match = pattern.search(file_name)
    groupdict = match.groupdict()
    return regularized_untangled_file_name(groupdict)


def to_float(key, decimal_separator='.'):
    return float(key.replace(decimal_separator, '.'))


def represents_float(key, decimal_separator='.'):
    try:
        float(key.replace(decimal_separator, '.'))
        return True
    except ValueError:
        return False


def represents_float_list(keys, decimal_separator='.'):
    return all(represents_float(key, decimal_separator) for key in keys)


def split_file(spamreader, decimal_separator):
    file_data = {'data': [], 'header': []}
    for row in spamreader:
        if represents_float_list(row, decimal_separator=decimal_separator):
            file_data['data'].append(row)
        else:
            file_data['header'].append(row)
    return file_data


unit = pint.UnitRegistry()


# Details
decimal_separator = ','
data_path = (
    pathlib.Path(__file__).parent.parent / 'Calibration' / 'Calibration Data'
)

# Containers
reference_temperature = {}
measured_temperature = {}
temperature_data = {}
file_data = {}
temperature_pairs = {}

# Read data
for data_file in data_path.iterdir():
    file_name = data_file.name

    # Reference temperature
    details = untangle_file_name(file_name)
    reference_temperature[file_name] = float(details['reference_temperature'])

    # Temperature data
    temperature_data[file_name] = []
    with open(data_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        file_data[file_name] = split_file(
            spamreader, decimal_separator=decimal_separator
        )
        for string_temperature_pair in file_data[file_name]['data']:
            time_instant = to_float(
                string_temperature_pair[0], decimal_separator=decimal_separator
            )
            temperature = to_float(
                string_temperature_pair[1], decimal_separator=decimal_separator
            )
            temperature_data[file_name].append(temperature)
    measured_temperature[file_name] = np.mean(
        np.array(temperature_data[file_name])
    )
    temperature_pairs[file_name] = (
        reference_temperature[file_name],
        measured_temperature[file_name],
    )

# 2-degree polynomial fit
x = np.array([])
y = np.array([])
for temperature_pair in temperature_pairs.values():
    x = np.append(x, temperature_pair[0])
    y = np.append(y, temperature_pair[1])


def fit_polynomial(x, y, degree):
    model = Pipeline(
        [
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression(fit_intercept=False)),
        ]
    )
    model.fit(x[:, np.newaxis], y)
    return model


model = fit_polynomial(x, y, 3)
coefficients = model.named_steps['linear'].coef_

with open(
    pathlib.Path(__file__).parent / 'coefficients.csv', 'w'
) as coeff_file:
    coeff_file.writelines('%s\n' % item for item in coefficients)
