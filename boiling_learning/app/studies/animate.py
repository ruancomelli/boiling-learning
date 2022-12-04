from pathlib import Path

import typer

from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.visualization.video import save_as_video

PREFETCH_BUFFER_SIZE = 256


app = typer.Typer()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    each: int = typer.Option(60),
    fps: int = typer.Option(30),
) -> None:
    for index, dataset in enumerate(boiling_datasets(direct_visualization=direct), start=1):
        for subset_name, subset in zip(('train', 'val', 'test'), dataset()):
            path = (
                _animations_path()
                / f"boiling-case-{index}-{'direct' if direct else 'indirect'}-{subset_name}.mp4"
            )

            if not path.is_file():
                save_as_video(
                    path,
                    subset[::each].prefetch(PREFETCH_BUFFER_SIZE),
                    display_data={'index': 'Index', 'Flux [W/cm**2]': 'Flux [W/cm²]'},
                    fps=fps,
                )


@app.command()
def condensation(
    fps: int = typer.Option(30),
) -> None:
    raise NotImplementedError


def _animations_path() -> Path:
    return studies_path() / 'animations'
