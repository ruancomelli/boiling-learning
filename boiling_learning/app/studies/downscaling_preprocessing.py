import functools
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import typer

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import (
    default_boiling_preprocessors,
    default_condensation_preprocessors,
)
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.datasets.raw.condensation import condensation_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.image_datasets import Image
from boiling_learning.preprocessing.image import (
    downscaler,
    normalized_mutual_information,
    retained_variance,
    shannon_cross_entropy_ratio,
    structural_similarity_ratio,
)
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()

BOILING_DOWNSCALING_INDEX = 3
CONDENSATION_DOWNSCALING_INDEX = 3


def pixels_in_image(sample_frame: Image, downscaled_frame: Image) -> int:
    return functools.reduce(operator.mul, downscaled_frame.shape, 1)


METRICS = (
    (retained_variance, 'linear'),
    (shannon_cross_entropy_ratio, 'linear'),
    # (shannon_entropy_ratio, 'linear'),
    (structural_similarity_ratio, 'linear'),
    (normalized_mutual_information, 'log'),
    (pixels_in_image, 'log'),
)


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    factors: list[int] = typer.Option(list(range(1, 10)) + list(range(10, 100, 10))),
) -> None:
    sample_frames: list[Image] = []

    preprocessors = default_boiling_preprocessors(direct_visualization=direct)[
        :BOILING_DOWNSCALING_INDEX
    ]

    for case in boiling_cases():
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
        )

        ds_train, _, _ = dataset()
        sample_frame, _ = ds_train[0]
        sample_frames.append(sample_frame)

    sns.set_style('whitegrid')

    NROWS = len(METRICS)
    NCOLS = len(sample_frames)

    f, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(NROWS * 4, NCOLS * 6),
        sharex='col',
        sharey='row',
    )

    x = [factor**2 for factor in factors]
    preferred_factor = 4
    for col, sample_frame in enumerate(sample_frames):
        downscaled_frames = [downscaler(factor)(sample_frame) for factor in factors]

        for row, (metric, y_scale) in enumerate(METRICS):
            ax = axes[row, col]

            y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

            ax.scatter(x, y, s=20, color='k')
            ax.scatter(
                x[0],
                y[0],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )
            ax.scatter(
                x[preferred_factor],
                y[preferred_factor],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )

            ax.set_xscale('log')
            ax.set_yscale(y_scale)

            if not row:
                ax.set_title(f'Dataset {col}')
            if not col:
                ax.set_ylabel(' '.join(metric.__name__.split('_')).title())

            ax.xaxis.grid(True, which='minor')

    output_path = resolve(
        _downscaling_study_path() / f"boiling1d-{'direct' if direct else 'indirect'}.png",
        parents=True,
    )
    f.savefig(str(output_path))


@app.command()
def condensation(
    factors: list[int] = typer.Option(list(range(1, 10))),
) -> None:
    sample_frames: list[Image] = []

    preprocessors = default_condensation_preprocessors()[:CONDENSATION_DOWNSCALING_INDEX]

    for case in condensation_datasets():
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='condensation',
        )

        ds_train, _, _ = dataset()
        sample_frame, _ = ds_train[0]
        sample_frames.append(sample_frame)

    sns.set_style('whitegrid')

    NROWS = len(METRICS)
    NCOLS = len(sample_frames)

    f, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=(NROWS * 4, NCOLS * 4),
        sharex='col',
        sharey='row',
    )

    x = [factor**2 for factor in factors]
    preferred_factor = 4
    for col, sample_frame in enumerate(sample_frames):
        downscaled_frames = [downscaler(factor)(sample_frame) for factor in factors]

        for row, (metric, y_scale) in enumerate(METRICS):
            ax = axes[row, col]

            y = [metric(sample_frame, downscaled_frame) for downscaled_frame in downscaled_frames]

            ax.scatter(x, y, s=20, color='k')
            ax.scatter(
                x[0],
                y[0],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )
            ax.scatter(
                x[preferred_factor],
                y[preferred_factor],
                facecolors='none',
                edgecolors='k',
                marker='$\\odot$',
                s=100,
            )

            ax.set_xscale('log')
            ax.set_yscale(y_scale)

            if not row:
                ax.set_title(f'Dataset {col}')
            if not col:
                ax.set_ylabel(' '.join(metric.__name__.split('_')).title())

            ax.xaxis.grid(True, which='minor')

    output_path = resolve(
        _downscaling_study_path() / 'condensation.png',
        parents=True,
    )
    f.savefig(str(output_path))


def _downscaling_study_path() -> Path:
    return studies_path() / 'downscaling-preprocessing'
