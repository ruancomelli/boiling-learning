import numpy as np

def smooth_fourier(y, min_value=None, max_value=None, min_balance=None, max_balance=None):
    from scipy.fftpack import rfft, irfft
    
    w = rfft(y)
    spectrum = w**2
    
    cutoff_idx = np.full(spectrum.shape, False)
            
    if min_value is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum < min_value)
    if max_value is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum > max_value)
    if min_balance is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum < (spectrum.max() * min_balance))
    if max_balance is not None:
        cutoff_idx = np.logical_or(cutoff_idx, spectrum > (spectrum.max() * max_balance))
        
    w2 = w.copy()
    w2[cutoff_idx] = 0

    return irfft(w2)

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
    return np.array(list(moving_average(np.pad(y, size//2, mode=mode), size=size)))

# def smooth_move(transform, y, size=3, pad_mode=None):
# TODO: implement this
    
def smooth_mean(y, size=3, **kwargs):
    from scipy.ndimage.filters import convolve
    
    kernel = np.full((size, *y.shape[1:]), 1.0) / size
    return np.array(convolve(y, kernel, **kwargs))