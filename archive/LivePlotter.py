import matplotlib.pyplot as plt
import numpy as np


class LivePlotter:
    def __init__(self, ax=None, *args, pause_time=1.0, **kwargs):
        self.config(*args, **kwargs)

        self.pause_time = 1.0
        self.lines = {}
        
        if ax is None:
            self.fig = plt.figure()
        else:
            self.ax = ax

    def plot(self, *args, i=-1, **kwargs):
        self.lines[i] = self.ax.plot(*args, **kwargs)[0]

    def config(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def update_limits(self, x, y):
        if np.min(x) <= self.ax.get_ylim()[0] or np.max(x) >= self.ax.get_ylim()[1]:
            self.ax.set_xlim(([np.min(x)-np.std(x), np.max(x)+np.std(x)]))

        if np.min(y) <= self.ax.get_ylim()[0] or np.max(y) >= self.ax.get_ylim()[1]:
            self.ax.set_ylim(([np.min(y)-np.std(y),np.max(y)+np.std(y)]))

    def update(self, x, y, i=-1):
        # self.ax.plot(x, y, *self.args, **self.kwargs)
        self.lines[i].set_xdata(x)
        self.lines[i].set_ydata(y)
        self.update_limits(x, y)

        plt.pause(self.pause_time)
