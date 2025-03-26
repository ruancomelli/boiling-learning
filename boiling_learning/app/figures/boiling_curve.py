import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from matplotlib.lines import Line2D
from rich.console import Console

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.displaying import glossary, units

app = typer.Typer()
console = Console()

NATURAL_CONVECTION = "Natural convection"
PARTIAL_NUCLEATE_BOILING = "Partial nucleate boiling"
FULLY_DEVELOPED_NUCLEATE_BOILING = "Fully; developed nucleate boiling"
FILM_BOILING = "Film boiling"

CENGEL_BOILING_CURVE = tuple(
    (temperature, heat_flux / 1e4, regime)  # W/m^2 -> W/cm^2
    for temperature, heat_flux, regime in (
        (0.985416453985273, 1045.71917377561, NATURAL_CONVECTION),
        (1.0772457814437986, 1115.894983689327, NATURAL_CONVECTION),
        (1.1776325318552103, 1195.9956020922273, NATURAL_CONVECTION),
        (1.2873741572930586, 1282.8122012003425, NATURAL_CONVECTION),
        (1.4073424230690168, 1376.9679226684214, NATURAL_CONVECTION),
        (1.5384903328604755, 1479.148598548596, NATURAL_CONVECTION),
        (1.681859699179309, 1591.3080999480271, NATURAL_CONVECTION),
        (1.8385894193200913, 1715.8466304815677, NATURAL_CONVECTION),
        (2.00992452253022, 1852.9220216409503, NATURAL_CONVECTION),
        (2.1972260602708373, 2003.9657900929456, NATURAL_CONVECTION),
        (2.4019819181347977, 2173.8643057144814, NATURAL_CONVECTION),
        (2.6258186353093564, 2363.5036950808003, NATURAL_CONVECTION),
        (2.870514325475014, 2581.330371039038, NATURAL_CONVECTION),
        (3.138012801781533, 2827.7425345045276, NATURAL_CONVECTION),
        (3.4304390181070703, 3114.0589618062395, NATURAL_CONVECTION),
        (3.7501159492626828, 3457.908198075388, NATURAL_CONVECTION),
        (4.099583044235129, 3860.03089552767, NATURAL_CONVECTION),
        (4.481616399056818, 4367.756884813692, NATURAL_CONVECTION),
        (4.899250809552097, 5043.844224229487, NATURAL_CONVECTION),
        (5.227228508609119, 5635.489916184172, NATURAL_CONVECTION),
        (5.425579487521521, 6193.347749476342, NATURAL_CONVECTION),
        (5.425579487521521, 6193.347749476342, PARTIAL_NUCLEATE_BOILING),
        (5.760818660922442, 7187.29335665532, PARTIAL_NUCLEATE_BOILING),
        (6.246855649998196, 8697.490026177835, PARTIAL_NUCLEATE_BOILING),
        (6.719253246136519, 10475.779182194812, PARTIAL_NUCLEATE_BOILING),
        (7.198163146979031, 12610.396875473427, PARTIAL_NUCLEATE_BOILING),
        (7.680040290680329, 15164.25981768366, PARTIAL_NUCLEATE_BOILING),
        (8.161057688476548, 18167.981839728225, PARTIAL_NUCLEATE_BOILING),
        (8.637151429549691, 21776.345400201295, PARTIAL_NUCLEATE_BOILING),
        (9.141019174795003, 26225.28731690343, PARTIAL_NUCLEATE_BOILING),
        (9.674281183504306, 31883.75123188217, PARTIAL_NUCLEATE_BOILING),
        (10.238652236456373, 38809.03163389831, PARTIAL_NUCLEATE_BOILING),
        (10.792151042141422, 45428.28788302393, PARTIAL_NUCLEATE_BOILING),
        (11.561353778835578, 60550.066507849624, PARTIAL_NUCLEATE_BOILING),
        (11.561353778835578, 60550.066507849624, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (12.28546616616843, 73672.70339504124, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (12.949615186294903, 88899.4199214251, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (13.649667925089345, 107421.49480176902, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (14.3294144940347, 127316.36956499137, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (15.104058814932076, 153266.80079592307, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (15.985188215198846, 185895.4562694879, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (16.917720290039874, 225470.3594136772, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (17.9046537306265, 271856.01904780755, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (18.94916216355543, 325078.9045582335, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (20.13598883754995, 392363.17583361035, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (21.483981733586496, 476244.88235124503, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (22.922215285913854, 572101.1570216488, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (24.555979552200395, 684251.1076813145, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (26.52013103730952, 818932.6095466323, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (28.874321960765013, 968118.6735595623, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (31.565072207277154, 1092985.8700822631, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (36.816592636734654, 1104576.843820719, FULLY_DEVELOPED_NUCLEATE_BOILING),
        (1528.2620204411828, 1104576.843820719, FILM_BOILING),
        (1630.5706958907592, 1341604.0231106388, FILM_BOILING),
        (1768.1410167345307, 1626223.0941266348, FILM_BOILING),
        (1901.8507761299193, 1962331.814477893, FILM_BOILING),
        (2045.6718895340414, 2372273.1333928118, FILM_BOILING),
        (2200.3689943252966, 2878437.3757922333, FILM_BOILING),
        (2366.764551030191, 3479757.9096287694, FILM_BOILING),
        (2545.7432160058106, 4230006.651202769, FILM_BOILING),
        (2727.189216995405, 5081402.003554879, FILM_BOILING),
        (2904.713379746224, 5963069.508715957, FILM_BOILING),
    )
)


@app.command()
def main() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    f, ax = plt.subplots(1, 1, figsize=(7, 4))
    data = pd.DataFrame(
        CENGEL_BOILING_CURVE,
        columns=["Wall superheat", "Heat flux", "Regime"],
    )
    sns.lineplot(
        ax=ax,
        data=data,
        x="Wall superheat",
        y="Heat flux",
        hue="Regime",
        legend=False,
        linewidth=3,
    )
    lines = [line for line in ax.get_children() if isinstance(line, Line2D)]
    colors = [line.get_color() for line in lines]
    regime_colors = dict(zip(data["Regime"].unique(), colors))

    onb_x, onb_y = [
        (temperature, heat_flux)
        for temperature, heat_flux, regime in CENGEL_BOILING_CURVE
        if regime == PARTIAL_NUCLEATE_BOILING
    ][0]
    ax.scatter(onb_x, onb_y, color=regime_colors[PARTIAL_NUCLEATE_BOILING])

    b_x, b_y = [
        (temperature, heat_flux)
        for temperature, heat_flux, regime in CENGEL_BOILING_CURVE
        if regime == FULLY_DEVELOPED_NUCLEATE_BOILING
    ][0]
    ax.scatter(b_x, b_y, color=regime_colors[FULLY_DEVELOPED_NUCLEATE_BOILING])

    dnb_x, dnb_y = [
        (temperature, heat_flux)
        for temperature, heat_flux, regime in CENGEL_BOILING_CURVE
        if regime == FULLY_DEVELOPED_NUCLEATE_BOILING
    ][-1]
    ax.scatter(dnb_x, dnb_y, color=regime_colors[FULLY_DEVELOPED_NUCLEATE_BOILING])

    burnout_x, burnout_y, _ = CENGEL_BOILING_CURVE[-1]
    ax.scatter(burnout_x, burnout_y, color=regime_colors[FILM_BOILING])

    start_film_line_x, start_film_line_y = [
        (temperature, heat_flux)
        for temperature, heat_flux, regime in CENGEL_BOILING_CURVE
        if regime == FILM_BOILING
    ][0]
    ax.plot(
        [dnb_x, start_film_line_x],
        [dnb_y, start_film_line_y],
        "--",
        color=regime_colors[FILM_BOILING],
    )

    for regime in (
        NATURAL_CONVECTION,
        PARTIAL_NUCLEATE_BOILING,
        FULLY_DEVELOPED_NUCLEATE_BOILING,
    ):
        temperature_range = data[data["Regime"] == regime]["Wall superheat"]
        start = temperature_range.min()
        end = temperature_range.max()

        ax.axvspan(start, end, color=regime_colors[regime], alpha=0.15)
    ax.axvspan(dnb_x, burnout_x, color=regime_colors[FILM_BOILING], alpha=0.15)

    ax.set(
        xlabel=f"Wall superheat, ${glossary['wall superheat']}$ [${units['temperature']}$]",
        ylabel=f"Heat flux, ${glossary['heat flux']}$ [${units['heat flux']}$]",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.minorticks_off()
    # grid.ax.set(
    #     xticks=[20, 25, 30],
    #     yticks=[5, 10, 20],
    # )
    ax.xaxis.set_major_formatter(lambda val, pos: int(val))
    # ax.yaxis.set_major_formatter(lambda val, pos: int(val))
    # ax.xaxis.set_major_locator(MaxNLocator('auto', integer=True))
    f.savefig(figures_path() / "boiling-curve.pdf", bbox_inches="tight")
