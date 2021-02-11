from pathlib import Path
import numpy as np
import pandas as pd
import time
import scipy
import scipy.fftpack
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt

def smooth_fourier(y, min_value=None, max_value=None, min_balance=None, max_balance=None):
    w = scipy.fftpack.rfft(y)
    spectrum = w**2
    
    cutoff_idx = np.full(spectrum.shape, False)
            
    if min_value is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum > min_value)
    if max_value is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum < max_value)
    if min_balance is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum > (spectrum.max()/min_balance))
    if max_balance is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum < (spectrum.max()/max_balance))
        
    w2 = w.copy()
    w2[cutoff_idx] = 0

    return scipy.fftpack.irfft(w2)

# Simple Moving Average (SMA):
# Source: https://docs.python.org/3/library/collections.html#deque-recipes
def moving_average(iterable, size=3):
    # moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
    # http://en.wikipedia.org/wiki/Moving_average
    from collections import deque
    from itertools import islice
    
    it = iter(iterable)
    d = deque(islice(it, size-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / size
    
def smooth_moving_average(y, size=3, mode='symmetric'):
    return np.array(moving_average(np.pad(y, size//2, mode=mode), size=size))
    
def smooth_mean(y, size=3, **kwargs):
    kernel = np.full((size, *y.shape[1:]), 1.0) / size
    return np.array(scipy.ndimage.filters.convolve(y, kernel, **kwargs))


filename = Path('.') / 'experiments' / 'Experiment Output 20-01-2020' / 'Experiment 0 -- 17-00.csv'

for chunk in pd.read_csv(
        filename,
        delimiter=',',
        chunksize=10000
    ):
    fig, ax = plt.subplots()
    
    plt.plot(chunk['Elapsed time'], chunk['Bulk Temperature'], '.', label='Bulk Temperature')
    plt.plot(chunk['Elapsed time'], chunk['Wire Temperature'], '.', label='Wire Temperature')
    plt.grid(True)
    
    formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: time.strftime('%M:%S', time.gmtime(x)))
    ax.xaxis.set_major_formatter(formatter)
    
    # plt.plot(chunk['Elapsed time'], smooth_fourier(chunk['Wire Temperature'], max_balance=10000), '-', label='FFT-Filtered Wire Temperature')
    size = 9
    plt.plot(chunk['Elapsed time'], smooth_mean(chunk['Wire Temperature'], size), '-', label=f'{size}-Mean-Filtered Wire Temperature')
    
    plt.legend()
    plt.xlabel("Elapsed time [min:s]")
    plt.ylabel("Temperature [°C]")
    plt.savefig('images/foo.png')
    break
data = chunk
print(data)

plt.plot(data['Elapsed time'], data['Wire Temperature'], 'r,')
plt.grid(True)
plt.title("Signal-Diagram")
plt.xlabel("Sample")
plt.ylabel("In-Phase")
plt.savefig('foo2.png')