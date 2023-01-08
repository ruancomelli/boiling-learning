from pathlib import Path

import numpy as np
import typer
from loguru import logger
from PIL import Image

from boiling_learning.app.datasets.bridged.boiling1d import DEFAULT_BOILING_HEAT_FLUX_TARGET
from boiling_learning.app.datasets.generators import sliceable_dataset_from_video_and_transformers
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


@app.command()
def boiling1d() -> None:
    for direct in False, True:
        direct_label = 'direct' if direct else 'indirect'
        logger.info(f'Generating {direct_label} images')
        for quality in 'original', 'cropped', 'final':
            logger.info(f'Quality: {quality}')
            preprocessors = (
                []
                if quality == 'original'
                else default_boiling_preprocessors(direct_visualization=direct)[:3]
                if quality == 'cropped'
                else default_boiling_preprocessors(direct_visualization=direct)
            )
            for case_index, case in enumerate(boiling_cases()):
                logger.info(f'Getting frames from dataset #{case_index}')
                for ev in sorted(case(), key=lambda ev: ev.name):
                    logger.debug(f'Getting example frame from video {ev.name}')

                    dataset = sliceable_dataset_from_video_and_transformers(
                        ev,
                        preprocessors,
                        experiment='boiling1d',
                    )

                    frame, targets = dataset[0]
                    power = targets['nominal_power']
                    heat_flux = targets[DEFAULT_BOILING_HEAT_FLUX_TARGET]

                    output_path = _example_frames_study_path() / (
                        'boiling1d'
                        f'-{direct_label}'
                        f'-{quality}'
                        f'-dataset-{case_index}'
                        f'-ev-{ev.name}'
                        f'-power-{power}'
                        f'-hf-{heat_flux:.2f}'
                        f'.png'
                    )

                    if quality == 'original':
                        im = Image.fromarray(frame)
                    else:
                        im = Image.fromarray((frame * 255).astype(np.uint8).squeeze())
                    im.save(output_path)


@app.command()
def condensation(
    number_of_columns: int = typer.Option(4),
) -> None:
    raise NotImplementedError


def _example_frames_study_path() -> Path:
    return resolve(studies_path() / 'example-frames', dir=True)
