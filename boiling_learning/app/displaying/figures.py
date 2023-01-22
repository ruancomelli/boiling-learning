from typing import Final

from matplotlib.figure import Figure

from boiling_learning.utils.pathutils import PathLike, resolve

DATASET_MARKER_STYLE: Final = (
    ('Large wire', 'o'),
    ('Small wire', '.'),
    ('Horizontal ribbon', '>'),
    ('Vertical ribbon', '^'),
)


def save_figure(
    figure: Figure,
    path: PathLike,
) -> None:
    figure.savefig(
        str(resolve(path, parents=True)),
        bbox_inches='tight',
    )
