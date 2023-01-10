from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.preprocessing import default_boiling_preprocessors
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.datasets.sliceable import features
from boiling_learning.preprocessing.image import structural_similarity_ratio
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


DATASET_MARKER_STYLE = (
    ('Large wire', 'k', 'o'),
    ('Small wire', 'r', '.'),
    ('Horizontal ribbon', 'g', '>'),
    ('Vertical ribbon', 'b', '^'),
)

DEFAULT_INDICES = list(range(1, 11)) + list(range(10, 110, 10)) + [200, 300]


@app.command()
def boiling1d(
    # direct: bool = typer.Option(..., '--direct/--indirect'),
    # indices: list[int] = typer.Option(list(range(1, 10)) + list(range(10, 100, 10))),
    indices: list[int] = typer.Option(DEFAULT_INDICES),
) -> None:
    indices = sorted(frozenset(indices) | {0})

    f, ax = plt.subplots(
        1,
        1,
        figsize=(6, 4),
    )

    metrics: list[tuple[int, float, str]] = []
    for case, (dataset_name, _, _) in zip(boiling_cases(), DATASET_MARKER_STYLE):
        preprocessors = [
            default_boiling_preprocessors(
                direct_visualization=False,
                downscale_factor=1,
            )[0]
        ]
        dataset = get_image_dataset(
            case(),
            transformers=preprocessors,
            experiment='boiling1d',
            shuffle=False,
        )

        ds_train, _, _ = dataset()
        frames = features(ds_train).fetch(indices)
        reference_frame = frames[0]
        metrics.extend(
            [
                (index, structural_similarity_ratio(reference_frame, frame), dataset_name)
                for index, frame in zip(indices, frames)
            ]
        )

    df = pd.DataFrame(
        metrics,
        columns=['Index', 'Structural similarity ratio', 'Dataset'],
    )

    sns.scatterplot(
        ax=ax,
        data=df,
        x='Index',
        y='Structural similarity ratio',
        hue='Dataset',
        markers=[marker for _, _, marker in DATASET_MARKER_STYLE],
    )
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.xaxis.grid(True, which='minor')

    f.savefig(_consecutive_frames_study_path() / 'boiling1d.pdf')


@app.command()
def condensation(
    indices: list[int] = typer.Option(DEFAULT_INDICES),
) -> None:
    raise NotImplementedError


def _consecutive_frames_study_path() -> Path:
    return resolve(studies_path() / 'consecutive-frames', dir=True)
