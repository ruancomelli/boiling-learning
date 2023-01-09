from collections.abc import Iterator
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from skimage.io import imsave

from boiling_learning.app.constants import figures_path
from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import (
    RECOMMENDED_DOWNSCALE_FACTOR,
    default_boiling_preprocessors,
)
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.preprocessing.image import VideoFrameOrFrames
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils.iterutils import accumulate_parts
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()
console = Console()

VISUALIZATION_STEP = 5


@app.command()
def boiling1d(
    frame: int = typer.Option(0),
    ds: int = typer.Option(RECOMMENDED_DOWNSCALE_FACTOR),
) -> None:
    table = Table(
        'Dataset',
        'Visualization',
        'Step',
        'Image shape',
        title='Shape summary',
    )

    for case_index, case in enumerate(boiling_cases()):
        for direct in False, True:
            direct_label = 'direct' if direct else 'indirect'
            for step, preprocessors in enumerate(
                iter_preprocessors(
                    direct_visualization=direct,
                    downscale_factor=ds,
                )
            ):
                dataset = get_image_dataset(
                    case(),
                    transformers=preprocessors,
                    experiment='boiling1d',
                    shuffle=False,
                )

                ds_train, _, _ = dataset()
                sample_frame, target = ds_train[frame]

                output_path = _preprocessing_study_path() / (
                    'boiling1d'
                    f'-{direct_label}'
                    f'-ds-{ds}'
                    f'-case-{case_index}'
                    f'-frame-{frame}'
                    f'-step-{step}'
                    '.png'
                )
                imsave(output_path, sample_frame)

                if case_index == 0 and ds == RECOMMENDED_DOWNSCALE_FACTOR:
                    imsave(
                        _preprocessing_figures_path()
                        / (
                            f'step-{step}'
                            + (f'-{direct_label}' if step >= VISUALIZATION_STEP else '')
                            + f'-{target["nominal_power"]}W.png'
                        ),
                        sample_frame,
                    )

                table.add_row(
                    str(case_index),
                    direct_label,
                    str(step),
                    str(sample_frame.shape),
                )

    console.print(table)


def iter_preprocessors(
    direct_visualization: bool, downscale_factor: int
) -> Iterator[
    list[
        list[
            Transformer[VideoFrameOrFrames, VideoFrameOrFrames]
            | dict[str, Transformer[VideoFrameOrFrames, VideoFrameOrFrames]]
        ]
    ]
]:
    for preprocessor_groups in accumulate_parts(
        default_boiling_preprocessors(
            direct_visualization=direct_visualization,
            downscale_factor=downscale_factor,
        )
    ):
        if not preprocessor_groups:
            continue

        for last_preprocessors in accumulate_parts(preprocessor_groups[-1]):
            if not last_preprocessors:
                continue

            yield [*preprocessor_groups[:-1], last_preprocessors]


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _preprocessing_study_path() -> Path:
    return resolve(studies_path() / 'preprocessing', dir=True)


def _preprocessing_figures_path() -> Path:
    return resolve(figures_path() / 'machine-learning' / 'preprocessing', dir=True)
