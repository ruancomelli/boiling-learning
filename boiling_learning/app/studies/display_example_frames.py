import math
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from loguru import logger

from boiling_learning.app.datasets.generators import compile_transformers
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import BOILING_CASES
from boiling_learning.app.paths import STUDIES_PATH
from boiling_learning.image_datasets import Image
from boiling_learning.management.allocators import JSONAllocator
from boiling_learning.management.cacher import cache
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.utils.pathutils import resolve

EXAMPLE_FRAMES_STUDY_PATH = STUDIES_PATH / 'example-frames'


def display_example_frames(
    number_of_columns: int = typer.Option(4),
    direct: bool = typer.Option(..., '--direct/--indirect'),
    output_path: Path = typer.Option(Path('example.png'), '--output-path'),
) -> None:
    logger.debug(f"Displaying {'directly' if direct else 'indirectly'} visualized boiling frames")

    total_examples_count = sum(len(case()) for case in BOILING_CASES)
    number_of_rows = math.ceil(total_examples_count / number_of_columns)

    preprocessors = default_boiling_preprocessors(direct_visualization=direct)

    fig, axs = plt.subplots(
        number_of_rows, number_of_columns, figsize=(number_of_columns * 8, number_of_rows * 8)
    )

    index = 0
    for case in BOILING_CASES:
        for ev in sorted(case(), key=lambda ev: ev.name):
            try:
                transformer = compile_transformers(preprocessors, ev)
            except KeyError:
                # some experiment videos have no transformers associated
                continue

            logger.debug(f'Getting example frame from case {case().name} and video {ev.name}')
            frame = _get_first_frame(ev)
            logger.debug('Transforming frame')
            frame = transformer()(frame)

            logger.debug('Showing frame')
            col = index % number_of_columns
            row = index // number_of_columns

            axs[row, col].imshow(frame.squeeze(), cmap='gray')
            axs[row, col].set_title(f'{case().name} - {ev.name}')
            axs[row, col].grid(False)

            index += 1

    if not output_path.is_absolute():
        output_path = EXAMPLE_FRAMES_STUDY_PATH / 'frames' / output_path

    logger.debug(f'Saving figure to {output_path}')
    fig.savefig(resolve(output_path, parents=True))


@cache(JSONAllocator(EXAMPLE_FRAMES_STUDY_PATH / 'cache'))
def _get_first_frame(ev: ExperimentVideo) -> Image:
    return ev.video[0]
