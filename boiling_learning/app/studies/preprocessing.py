from pathlib import Path

import typer
from skimage.io import imsave

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.utils.iterutils import accumulate_parts
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    frame: int = typer.Option(0),
) -> None:
    for case_index, case in enumerate(boiling_cases(), start=1):
        for step, preprocessors in enumerate(
            accumulate_parts(default_boiling_preprocessors(direct_visualization=direct))
        ):
            dataset = get_image_dataset(
                case(),
                transformers=preprocessors,
                experiment='boiling1d',
            )

            ds_train, _, _ = dataset()
            sample_frame, _ = ds_train[frame]

            output_path = resolve(
                _preprocessing_study_path()
                / (
                    'boiling1d'
                    f"-{'direct' if direct else 'indirect'}"
                    f'-case-{case_index}'
                    f'-frame-{frame}'
                    f'-step-{step}'
                    '.png'
                ),
                parents=True,
            )
            imsave(output_path, sample_frame)


@app.command()
def condensation() -> None:
    raise NotImplementedError


def _preprocessing_study_path() -> Path:
    return studies_path() / 'preprocessing'
