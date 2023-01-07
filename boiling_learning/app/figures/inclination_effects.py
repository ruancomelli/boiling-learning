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
    # extracted from rohsenow1998handbook fig. 15.40
    (9.502495065177607, 16714.80313245148, 'Horizontal'),
    (9.685866562989549, 18109.705688884806, 'Horizontal'),
    (9.910589239605887, 20647.78254473832, 'Horizontal'),
    (10.257486843918356, 24683.404670686512, 'Horizontal'),
    (10.576020768301124, 29293.55985757924, 'Horizontal'),
    (11.157441439325249, 39063.58989444113, 'Horizontal'),
    (11.681177025719538, 47730.66315430856, 'Horizontal'),
    (11.997969847960295, 55219.008921485125, 'Horizontal'),
    (12.417931456321355, 65056.566576956626, 'Horizontal'),
    (12.951231968556847, 77771.91356142456, 'Horizontal'),
    (13.251714686806775, 87388.78398927077, 'Horizontal'),
    (15.679036523045706, 168984.97868124588, 'Horizontal'),
    (16.289997952029896, 191967.56144502945, 'Horizontal'),
    (16.73178172212064, 207987.85436083775, 'Horizontal'),
    (17.85521190573186, 237137.37056616554, 'Horizontal'),
    (19.200306381968158, 309390.04035341786, 'Horizontal'),
    (20.333430037616814, 378034.6824066051, 'Horizontal'),
    (21.287890875441516, 445383.5547710245, 'Horizontal'),
    (5.314867622715202, 8518.825982683438, 'Vertical'),
    (5.585661424518415, 9963.632980336315, 'Vertical'),
    (5.825543252610909, 11318.723955792093, 'Vertical'),
    (6.122356102323547, 12999.422244132475, 'Vertical'),
    (6.4097423979005566, 15093.760979789195, 'Vertical'),
    (6.736320336566123, 17335.00816829131, 'Vertical'),
    (7.106652049799782, 20423.3312954976, 'Vertical'),
    (7.497342886557023, 23541.57109667816, 'Vertical'),
    (8.00074063794566, 29080.883389047438, 'Vertical'),
    (8.472911669970447, 33277.592456965496, 'Vertical'),
    (9.146101038546519, 42633.09240147986, 'Vertical'),
    (9.502495065177607, 47557.08095775932, 'Vertical'),
    (9.835108262375577, 53049.77476004904, 'Vertical'),
    (10.61652680878764, 64819.97523336089, 'Vertical'),
    (11.372748759607436, 78913.6077401726, 'Vertical'),
    (15.619215105021661, 193371.47282195062, 'Vertical'),
    (16.289997952029896, 218871.8321796612, 'Vertical'),
    (18.269472311335615, 311652.694492943, 'Vertical'),
    (20.48948192120967, 450278.29470415326, 'Vertical'),
]


@app.command()
def main() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    data = pd.DataFrame(DATA, columns=['Excess temperature', 'Heat flux', 'Inclination'])
    data['Heat flux'] /= 1e4  # W/m^2 -> W/cm^2

    grid = sns.lmplot(
        data=data,
        x='Excess temperature',
        y='Heat flux',
        hue='Inclination',
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
        xscale='log',
        yscale='log',
        xlabel=f'Excess temperature, ${glossary["excess temperature"]}$ [${units["temperature"]}$]',
        ylabel=f'Heat flux, ${glossary["heat flux"]}$ [${units["heat flux"]}$]',
    )
    grid.ax.minorticks_off()
    grid.ax.set(xticks=[5, 10, 20])
    grid.ax.xaxis.set_major_formatter(lambda val, pos: int(val))
    grid.ax.yaxis.set_major_formatter(lambda val, pos: int(val))
    # grid.ax.xaxis.set_major_locator(MaxNLocator('auto', integer=True))
    grid.figure.savefig(figures_path() / 'surface-inclination-effects.pdf')
