from matplotlib.figure import Figure

from boiling_learning.utils.pathutils import PathLike, resolve


def save_figure(
    figure: Figure,
    path: PathLike,
) -> None:
    figure.savefig(
        str(resolve(path, parents=True)),
        bbox_inches='tight',
    )
