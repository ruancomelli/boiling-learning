import matplotlib as mpl
import matplotlib.pyplot as plt
import modin.pandas as pd
import seaborn as sns
from pathlib2 import Path

STEADY_STATE_DIR = Path(__file__).parent.parent / 'Selected Experiments'
STEADY_STATE_PATH = STEADY_STATE_DIR / 'SteadyState.csv'

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------


def figsize(height=None, width=None, ratio=None, textwidth=6.29707):
    height = width * ratio if height is None else height
    width = height / ratio if width is None else width

    return (width * textwidth, height * textwidth)


sns.set()
sns.set_style('white')
plt.rc('font', **{'family': 'serif', 'serif': ['Fourier']})
## for Palatino and other serif fonts use:
# plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

mpl.use('pgf')
pgf_with_latex = {  # setup matplotlib to use latex for output
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
    'figure.figsize': figsize(
        width=0.3, ratio=3 / 4
    ),  # default fig size of 0.9 textwidth
    'pgf.preamble': [
        r'\usepackage[utf8x]{inputenc}',  # use utf8 fonts
        r'\usepackage[T1]{fontenc}',  # plots will be generated
        r'\usepackage{fourier}',  # plots will be generated
        r'\usepackage[detect-all]{siunitx}',
    ],  # using this preamble
}
# }}}
mpl.rcParams.update(pgf_with_latex)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
data = pd.read_csv(STEADY_STATE_PATH)

elapsed_time_min = data['Elapsed time'] / 60

plt.plot(elapsed_time_min, data['Temperature'])
plt.xlabel('Elapsed Time [$\si{\min}$]')
# plt.show()
plt.savefig(STEADY_STATE_DIR / 'file.pdf')
