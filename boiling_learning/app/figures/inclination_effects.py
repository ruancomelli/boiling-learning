from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console
from tikzplotlib import save as export_latex

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.paths import studies_path
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

DATA = [
    # extracted from rohsenow1998handbook fig. 15.40
    (3.5487339990560107, 4865.377746586326, 'horizontal'),
    (4.323652799352948, 6034.9662630345865, 'horizontal'),
    (4.999999999999998, 7433.025867296345, 'horizontal'),
    (5.366861406492776, 8061.9800252866, 'horizontal'),
    (6.276172843451533, 9894.612096431456, 'horizontal'),
    (7.12396675077773, 11973.492841736817, 'horizontal'),
    (7.477589411741044, 10884.53338955715, 'horizontal'),
    (8.11647234314082, 13171.398873990267, 'horizontal'),
    (8.14677559962554, 16984.83568211756, 'horizontal'),
    (8.942249331776564, 15063.091369144018, 'horizontal'),
    (9.00914665389615, 22450.550461363116, 'horizontal'),
    (9.598362565189497, 27360.029695669888, 'horizontal'),
    (9.962803374029209, 22931.34075230829, 'horizontal'),
    (10.037335501439049, 19424.2186587222, 'horizontal'),
    (10.150180453237327, 33225.566361201265, 'horizontal'),
    (11.017408738333966, 36420.81779596557, 'horizontal'),
    (11.05854278631442, 41946.85323287803, 'horizontal'),
    (11.099830410325632, 30633.47880344075, 'horizontal'),
    (11.607471699555939, 38402.17551631442, 'horizontal'),
    (12.04819649464881, 51482.122270640975, 'horizontal'),
    (12.45913837826078, 59924.906069581615, 'horizontal'),
    (12.884096735772527, 70994.87418245163, 'horizontal'),
    (12.884096735772534, 81478.4849240751, 'horizontal'),
    (12.884096735772534, 87750.8907711747, 'horizontal'),
    (13.624814228141767, 95512.75005622313, 'horizontal'),
    (13.777991483660905, 105068.46625046853, 'horizontal'),
    (14.354522857502525, 116399.447859894, 'horizontal'),
    (14.46190961265135, 129866.43911425766, 'horizontal'),
    (14.899550604179744, 222925.37670808815, 'horizontal'),
    (15.179776376877648, 201936.4531341885, 'horizontal'),
    (15.756138224351371, 245228.28004888655, 'horizontal'),
    (16.66197178290893, 283435.33258768346, 'horizontal'),
    (16.84929434500095, 332255.66361201194, 'horizontal'),
    (17.166190246942985, 267863.8523328381, 'horizontal'),
    (18.702430976316933, 416516.20412386366, 'horizontal'),
    (20.000000000000004, 496957.2358044302, 'horizontal'),
    (21.708793827049703, 603496.6263034581, 'horizontal'),
    (3.7248876510449325, 5409.147626295346, 'vertical'),
    (4.471124665888282, 6592.018308789252, 'vertical'),
    (4.675607744079237, 7646.022696056304, 'vertical'),
    (5.0750902266186575, 9187.34836129424, 'vertical'),
    (5.825404388595091, 10432.896716424038, 'vertical'),
    (5.868984470004351, 12273.178630307642, 'vertical'),
    (6.252827598072934, 13887.946563637823, 'vertical'),
    (6.761821657908654, 19492.937962067375, 'vertical'),
    (6.863371002038123, 18618.24111432401, 'vertical'),
    (7.36695218985601, 21519.00001657284, 'vertical'),
    (7.561656365228274, 26976.25107727501, 'vertical'),
    (7.703873473254861, 24784.021612866425, 'vertical'),
    (7.9370052598409835, 30850.61341330876, 'vertical'),
    (8.269124244822997, 30850.61341330876, 'vertical'),
    (8.551168905573538, 36420.81779596557, 'vertical'),
    (8.615140539504878, 34663.89021911049, 'vertical'),
    (8.647305598705898, 39923.35162948195, 'vertical'),
    (9.281778137568722, 50581.03713543593, 'vertical'),
    (9.527090026622664, 43454.70538341262, 'vertical'),
    (10.074810396844857, 55838.34212543075, 'vertical'),
    (10.226114356012681, 59503.13905553648, 'vertical'),
    (11.05854278631442, 68289.79802136263, 'vertical'),
    (11.05854278631442, 81766.74077724369, 'vertical'),
    (11.058542786314426, 73807.10305520911, 'vertical'),
    (11.650808777190187, 85005.6617823587, 'vertical'),
    (12.274794951342008, 98249.7125303653, 'vertical'),
    (12.505655196145867, 108461.61669849057, 'vertical'),
    (12.836172243047319, 129408.61529923184, 'vertical'),
    (13.423223647020224, 152235.28949231518, 'vertical'),
    (14.247933501555448, 173485.74829775206, 'vertical'),
    (14.955178823482083, 222925.37670808815, 'vertical'),
    (15.23645083315399, 200515.17206006584, 'vertical'),
    (17.102337811147077, 266919.54022557713, 'vertical'),
    (17.751697510013717, 295705.1550672598, 'vertical'),
    (18.15308887507458, 361644.78644544125, 'vertical'),
    (20.68216331596931, 471316.7599041525, 'vertical'),
    (21.547595410677207, 546676.0670937458, 'vertical'),
    (22.617184628742102, 618601.6056728169, 'vertical'),
    (22.95685053259176, 692613.2531994942, 'vertical'),
]


@app.command()
def main() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        modin_engine='ray',
        require_gpu=True,
    )

    data = pd.DataFrame(
        DATA, columns=['Excess temperature', 'Heat flux', 'Inclination']
    ).sort_values(by='Excess temperature')

    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.regplot(
        ax=ax,
        data=data[data['Inclination'] == 'horizontal'],
        x='Excess temperature',
        y='Heat flux',
        # hue='Inclination',
        order=3,
        ci=None,
        truncate=True,
    )
    sns.regplot(
        ax=ax,
        data=data[data['Inclination'] == 'vertical'],
        x='Excess temperature',
        y='Heat flux',
        # hue='Inclination',
        order=3,
        ci=None,
        truncate=True,
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    f.savefig(_inclination_effects_path() / 'figure.pdf')
    export_latex(
        str(figures_path() / 'surface-inclination-effects.tex'),
        f,
        flavor='latex',
    )


def _inclination_effects_path() -> Path:
    return resolve(studies_path() / 'inclination-effects', dir=True)
