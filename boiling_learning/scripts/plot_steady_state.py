from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_STEADY_STATE_DIR = Path(__file__).parent.parent / 'Selected Experiments'


def main(
    data_path: Path = _STEADY_STATE_DIR / 'SteadyState.csv',
    output_path: Path = _STEADY_STATE_DIR / 'file.pdf',
) -> None:
    sns.set()
    sns.set_style('white')
    plt.rc('font', **{'family': 'serif', 'serif': ['Fourier']})
    # plt.rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rc('text', usetex=True)

    mpl.use('pgf')
    mpl.rcParams.update(
        {  # setup matplotlib to use latex for output
            'pgf.texsystem': 'pdflatex',  # change this if using xetex or lautex
            'text.usetex': True,  # use LaTeX to write all text
            'font.family': 'serif',
            'font.serif': [],  # blank entries should cause plots
            'font.sans-serif': [],  # to inherit fonts from the document
            'font.monospace': [],
            'axes.labelsize': 12,  # LaTeX default is 10pt font.
            'font.size': 12,
            'legend.fontsize': 10,  # Make the legend/label fonts
            'xtick.labelsize': 10,  # a little smaller
            'ytick.labelsize': 10,
            'figure.figsize': _figsize(width=0.3, ratio=0.75),  # default fig size of 0.9 textwidth
            'pgf.preamble': [
                r'\usepackage[utf8x]{inputenc}',  # use utf8 fonts
                r'\usepackage[T1]{fontenc}',  # plots will be generated
                r'\usepackage{fourier}',  # plots will be generated
                r'\usepackage[detect-all]{siunitx}',
            ],  # using this preamble
        }
    )

    data = pd.read_csv(data_path)

    elapsed_time_min = data['Elapsed time'] / 60

    plt.plot(elapsed_time_min, data['Temperature'])
    plt.xlabel('Elapsed Time [$\si{\min}$]')
    # plt.show()
    plt.savefig(output_path)


def _figsize(
    height: Optional[float] = None,
    width: Optional[float] = None,
    ratio: Optional[float] = None,
    textwidth: float = 6.29707,
) -> tuple[float, float]:
    if height is None:
        if width is None or ratio is None:
            raise ValueError('if `height` is omitted, both `width` and `ratio` must be given.')
        height = width * ratio

    if width is None:
        if height is None or ratio is None:
            raise ValueError('if `width` is omitted, both `height` and `ratio` must be given.')
        width = height / ratio

    return (width * textwidth, height * textwidth)
