import functools
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from loguru import logger

from boiling_learning.app.datasets.generators import compile_transformers
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.image_datasets import Image
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


@app.command()
def boiling1d(
    number_of_columns: int = typer.Option(4),
    direct: bool = typer.Option(..., "--direct/--indirect"),
) -> None:
    logger.debug(
        f"Displaying {'directly' if direct else 'indirectly'} visualized boiling frames"
    )

    total_examples_count = sum(len(case()) for case in boiling_cases())
    number_of_rows = math.ceil(total_examples_count / number_of_columns)

    preprocessors = default_boiling_preprocessors(direct_visualization=direct)

    fig, axs = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(number_of_columns * 8, number_of_rows * 8),
    )

    index = 0
    for case_index, case in enumerate(boiling_cases()):
        for ev in sorted(case(), key=lambda ev: ev.name):
            try:
                transformer = compile_transformers(preprocessors, ev)
            except KeyError:
                # some experiment videos have no transformers associated
                continue

            logger.debug(
                f"Getting example frame from dataset {case_index} and video {ev.name}"
            )
            frame = _first_frame_getter()(ev)
            logger.debug("Transforming frame")
            frame = transformer()(frame)

            logger.debug("Showing frame")
            col = index % number_of_columns
            row = index // number_of_columns

            axs[row, col].imshow(frame.squeeze(), cmap="gray")
            axs[row, col].set_title(f"Dataset {case_index} - {ev.name}")
            axs[row, col].grid(False)

            index += 1

    output_path = resolve(
        _example_frames_figures_path()
        / f"boiling1d-{'direct' if direct else 'indirect'}.png",
        parents=True,
    )

    logger.debug(f"Saving figure to {output_path}")
    fig.savefig(resolve(output_path, parents=True))


@app.command()
def condensation(
    number_of_columns: int = typer.Option(4),
) -> None:
    raise NotImplementedError


@functools.cache
def _first_frame_getter() -> Callable[[ExperimentVideo], Image]:
    @cache(JSONAllocator(_frames_cache_path()))
    def _get_first_frame(ev: ExperimentVideo) -> Image:
        return ev.frames()[0]

    return _get_first_frame


def _example_frames_figures_path() -> Path:
    return _example_frames_study_path() / "frames"


def _frames_cache_path() -> Path:
    return _example_frames_study_path() / "cache"


def _example_frames_study_path() -> Path:
    return studies_path() / "example-frames-matrix"
