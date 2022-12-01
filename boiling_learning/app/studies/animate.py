import typer

from boiling_learning.app.datasets.preprocessed.boiling1d import (
    BOILING_DIRECT_DATASETS,
    BOILING_INDIRECT_DATASETS,
)
from boiling_learning.app.paths import STUDIES_PATH
from boiling_learning.visualization.video import save_as_video

ANIMATIONS_PATH = STUDIES_PATH / 'animations'
PREFETCH_BUFFER_SIZE = 256


def animate(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    each: int = typer.Option(60),
    fps: int = typer.Option(30),
) -> None:
    for index, dataset in enumerate(
        BOILING_DIRECT_DATASETS if direct else BOILING_INDIRECT_DATASETS, start=1
    ):
        for subset_name, subset in zip(('train', 'val', 'test'), dataset()):
            path = (
                ANIMATIONS_PATH
                / f"boiling-case-{index}-{'direct' if direct else 'indirect'}-{subset_name}.mp4"
            )

            if not path.is_file():
                save_as_video(
                    path,
                    subset[::each].prefetch(PREFETCH_BUFFER_SIZE),
                    display_data={'index': 'Index', 'Flux [W/cm**2]': 'Flux [W/cm²]'},
                    fps=fps,
                )
