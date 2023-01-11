import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.displaying import glossary, units

app = typer.Typer()
console = Console()

DATA = [
    # extracted from kim2006 fig. 3
    (23.102409638554217, 87966.10169491524, 'Small'),
    (23.82530120481928, 133389.83050847458, 'Small'),
    (24.03614457831325, 144463.27683615815, 'Small'),
    (24.487951807228914, 155536.7231638418, 'Small'),
    (25.240963855421693, 171355.9322033898, 'Small'),
    (26.26506024096386, 191920.90395480226, 'Small'),
    (26.74698795180723, 200734.4632768361, 'Small'),
    (27.259036144578314, 209096.04519774002, 'Small'),
    (28.072289156626514, 222203.38983050844, 'Small'),
    (28.674698795180728, 231468.9265536723, 'Small'),
    (29.096385542168672, 240508.47457627114, 'Small'),
    (20.993975903614462, 58587.57062146888, 'Intermediate'),
    (22.138554216867472, 71468.92655367227, 'Intermediate'),
    (22.921686746987948, 82994.35028248586, 'Intermediate'),
    (23.85542168674699, 98361.58192090396, 'Intermediate'),
    (24.759036144578307, 111694.91525423725, 'Intermediate'),
    (25.421686746987955, 129096.04519774011, 'Intermediate'),
    (26.445783132530128, 144237.28813559317, 'Intermediate'),
    (27.439759036144576, 156214.6892655367, 'Intermediate'),
    (28.192771084337355, 168644.06779661012, 'Intermediate'),
    (29.18674698795181, 181525.4237288135, 'Intermediate'),
    # (15.421686746987955, 24237.288135593175, 'Large'),
    # (16.32530120481926, 28531.073446327413, 'Large'),
    # (16.98795180722892, 32824.85875706212, 'Large'),
    # (17.710843373493976, 36892.65536723164, 'Large'),
    # (18.072289156626503, 41864.40677966102, 'Large'),
    # (18.373493975903617, 46384.18079096044, 'Large'),
    # (18.79518072289157, 51129.94350282481, 'Large'),
    # (19.548192771084334, 59717.51412429378, 'Large'),
    # (19.87951807228916, 64689.265536723164, 'Large'),
    (20.240963855421686, 69209.03954802259, 'Large'),
    (20.903614457831328, 78022.59887005645, 'Large'),
    (21.62650602409639, 87288.13559322033, 'Large'),
    (22.379518072289162, 96779.66101694916, 'Large'),
    (23.132530120481935, 106723.16384180787, 'Large'),
    (23.73493975903615, 116214.6892655367, 'Large'),
    (24.638554216867472, 125706.21468926553, 'Large'),
    (25.69277108433735, 136553.67231638415, 'Large'),
    (26.6566265060241, 145593.22033898302, 'Large'),
    (27.289156626506024, 150112.99435028245, 'Large'),
    (27.861445783132535, 154858.75706214688, 'Large'),
    (28.373493975903617, 158700.5649717514, 'Large'),
    (29.27710843373494, 165932.2033898305, 'Large'),
]


@app.command()
def main() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    data = pd.DataFrame(DATA, columns=['Excess temperature', 'Heat flux', 'Diameter'])
    data['Heat flux'] /= 1e4  # W/m^2 -> W/cm^2

    grid = sns.lmplot(
        data=data,
        x='Excess temperature',
        y='Heat flux',
        hue='Diameter',
        order=3,
        ci=None,
        scatter=False,
        facet_kws={
            'legend_out': False,
            'despine': False,
        },
        height=3.5,
        aspect=1,
    )
    grid.ax.set(
        # xscale='log',
        # yscale='log',
        xlabel=f'Excess temperature, ${glossary["excess temperature"]}$ [${units["temperature"]}$]',
        ylabel=f'Heat flux, ${glossary["heat flux"]}$ [${units["heat flux"]}$]',
    )
    grid.ax.minorticks_off()
    grid.ax.set(
        xticks=[20, 25, 30],
        yticks=[5, 10, 20],
    )
    grid.ax.xaxis.set_major_formatter(lambda val, pos: int(val))
    grid.ax.yaxis.set_major_formatter(lambda val, pos: int(val))
    # grid.ax.xaxis.set_major_locator(MaxNLocator('auto', integer=True))
    grid.figure.savefig(figures_path() / 'diameter-effects.pdf')
