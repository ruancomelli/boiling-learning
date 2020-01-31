
#%%
# Playing with annotations:
if True:
    def hi(arg: str) -> bool:
        return True
    
print(hi.__annotations__)
print(hi(1))

#%%
# Accessing member variables
class X:
    def __init__(self, value=0):
        self.value = value

def change_value(x):
    x.value += 1
    
def change(x):
    x = X(5)
    
x = X(3)
change_value(x)
print(x.value)

change(x)
print(x.value)

#%%
# time and datetime
import time
import datetime

now = time.time()
now_date = datetime.datetime.fromtimestamp(now)
print(now_date)

# Merging dicts
def merge_dicts(*dicts, inplace=False):
    if len(dicts):
        if inplace:
            for d in dicts:
                dicts[0].update(d)
            return dicts[0]
        else:
            return merge_dicts({}, *dicts, inplace=True)
    else:
        return {}
    
    
a = {
    'p': 0,
    'y': 1
}
b = {
    't': 0,
    'p': 3
}
c = {}
d = {
    'h': -1
}

print(f'{merge_dicts({})}')
print(f'{merge_dicts(a, b)}')
print(f'{merge_dicts(a, c, b, d, d)}')
print(f'{merge_dicts(a, d, inplace=True)}')
print(f'{merge_dicts(a, b)}')

#%%
import numpy as np

wire_diameter = 0.518e-3 # m
wire_length = 6.5e-2 # m
wire_surface_area = np.pi * wire_diameter * wire_length
range_power = np.array([5, 15, 35, 75, 100])
heat_fluxes = range_power / wire_surface_area
print('Heat fluxes:', heat_fluxes, 'W/m^2')
print('Heat fluxes:', heat_fluxes/100**2, 'W/cm^2')

#%%
# Relative paths    
def common_values(lhs, rhs):
    for lvalue, rvalue in zip(lhs, rhs):
        if lvalue == rvalue:
            yield lvalue
        else:
            return

def common_list(lhs, rhs):
    return list(common_values(lhs, rhs))

def common_path(lhs, rhs):
    resolved_lhs = lhs.resolve()
    resolved_rhs = rhs.resolve()
    return common_list(
        reversed(resolved_lhs.parents),
        reversed(resolved_rhs.parents))[-1]
    
# def relative_path(origin, destination):
#     common = common_path(origin, destination)
    
#     upwards_path = origin.relative_to(common)
#     upwards = Path('/'.join(['..'] * len(upwards_path.parts)))
    
#     downwards = destination.relative_to(common)
    
#     return upwards.joinpath(downwards)

def relative_path(origin, destination):
    from os.path import relpath
    return relpath(destination, start=origin)

from pathlib import Path
origin      = Path('middle-earth/gondor/minas-tirith/castle').resolve()
destination = Path('middle-earth/gondor/osgiliath/tower').absolute()

import os.path

# result = Path('../../also/in/another/place')
result = relative_path(origin, destination)

print(os.path.relpath(destination, start=origin))

def absolute_file_path_to_relative(start_file_path, destination_file_path):
    return (start_file_path.count("/") + start_file_path.count("\\")) * (".." + ((start_file_path.find("/") > -1) and "/" or "\\")) + destination_file_path

def relative_file_path(start, dest):
    count = len(start.parts)
    upwards = Path('/'.join(['..']*count))
    return upwards / dest

# result = absolute_file_path_to_relative(str(origin), str(destination))
print(absolute_file_path_to_relative(str(origin), str(destination)))

result = relative_file_path(origin, destination)

print(f'Origin: {origin}')
print(f'Destination: {destination}')
print(f'Result: {result}')
print(f'Got to: {origin.joinpath(result).resolve()}')
print(f'Works: {origin.joinpath(result).resolve() == destination}')
# print(f'Equal: {expected_result.resolve() == result.resolve()}')

#%%
import numpy as np
def relu(x, grad=False):
    numpy_x= np.array(x)
    if grad:
        return np.where(numpy_x <= 0, 0, 1)
    return np.maximum(0, numpy_x)

a=np.array([[1,2,3],[2,3,4]])
print(np.apply_along_axis(relu, 1, a))
print(np.apply_along_axis(lambda x: relu(x, grad=True), 1, a))

print(relu(a))
print(relu(a, True))

#%%

# def multi_split(*arrays, sizes=None, **options):
#     from sklearn.model_selection import train_test_split

#     if sizes is None:
#         return train_test_split(*arrays, **options)

#     if all(isinstance(size, int) for size in sizes):
#         split_arrays = []
#         for size in sizes:
#             if 'train_size' in previous_size:

#     elif all(isinstance(size, float) for size in sizes):
#         split_arrays = []
#         previous_size = 1
#         for size in sizes:
#             train_size = 
#             split_arrays.append(
#                 train_test_split(*arrays, train_size=train_size, **options)
#             )

#     else:
#         TypeError(f'there is no support for sizes={sizes}')

#%%
x = 3
try:
    print(x(6))
except TypeError:
    print('Success: TypeError raised')

#%%
try:
    from math import prod
except ImportError:
    def prod(iterable, start=1.0):
        from functools import reduce
        import operator
        
        return start*reduce(operator.mul, iterable, 1)
    
import numpy as np

dims = [4, 4, 4, 4]
a = np.arange(prod(dims))
a = a.reshape(dims)
print(a)

print('Sliced a')
print(
    a.__getitem__(
        tuple([slice(1, 4), slice(0, 2)])
    )
)

#%%
import sympy

wire_diameter = 0.518e-3 # m
wire_length = 6.5e-2 # m
wire_surface_area = np.pi * wire_diameter * wire_length