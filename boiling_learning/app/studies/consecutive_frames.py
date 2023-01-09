from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from boiling_learning.app.datasets.generators import get_image_dataset
from boiling_learning.app.datasets.raw.boiling1d import boiling_cases
from boiling_learning.app.paths import studies_path
from boiling_learning.datasets.sliceable import features
from boiling_learning.preprocessing.image import (
    VideoFrameOrFrames,
    grayscaler,
    image_dtype_converter,
    structural_similarity,
    structural_similarity_ratio,
)
from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.utils.pathutils import resolve

app = typer.Typer()


METRICS = (
    (structural_similarity, 'linear'),
    (structural_similarity_ratio, 'linear'),
)

DATASET_MARKER_STYLE = (
    ('Thick wire', 'k', 'o'),
    ('Thin wire', 'r', '.'),
    ('Horizontal ribbon', 'g', '>'),
    ('Vertical ribbon', 'b', '^'),
)

BOILING_PREPROCESSORS_COMPARE_INDEX = None

DEFAULT_INDICES = list(range(1, 501))


@app.command()
def boiling1d(
    # direct: bool = typer.Option(..., '--direct/--indirect'),
    # indices: list[int] = typer.Option(list(range(1, 10)) + list(range(10, 100, 10))),
    indices: list[int] = typer.Option(DEFAULT_INDICES),
) -> None:
    NROWS = len(METRICS)
    indices = sorted(frozenset(indices) | {0})

    f, axes = plt.subplots(
        NROWS,
        1,
        figsize=(6, NROWS * 4),
        sharex='col',
    )

    dataset = get_image_dataset(
        boiling_cases()[0](),
        transformers=_boiling_preprocessors(),
        experiment='boiling1d',
        shuffle=False,
    )

    ds_train, _, _ = dataset()
    frames = features(ds_train).fetch(indices)
    reference_frame = frames[0]

    for row, (metric, xscale) in enumerate(METRICS):
        metric_name = ' '.join(metric.__name__.split('_'))

        ax = axes[row]
        df = pd.DataFrame(
            [(index, metric(reference_frame, frame)) for index, frame in zip(indices, frames)],
            columns=['Index', metric_name],
        )

        sns.scatterplot(ax=ax, data=df, x='Index', y=metric_name)
        ax.set_xscale(xscale)
        ax.set_yscale('linear')
        ax.xaxis.grid(True, which='minor')

    f.savefig(_consecutive_frames_study_path() / 'boiling1d.pdf')


def _boiling_preprocessors() -> list[
    list[
        Transformer[VideoFrameOrFrames, VideoFrameOrFrames]
        | dict[str, Transformer[VideoFrameOrFrames, VideoFrameOrFrames]]
    ]
]:
    return [
        [image_dtype_converter('float32'), grayscaler()],
    ]


@app.command()
def condensation(
    indices: list[int] = typer.Option(DEFAULT_INDICES),
) -> None:
    raise NotImplementedError


def _consecutive_frames_study_path() -> Path:
    return resolve(studies_path() / 'consecutive-frames', dir=True)
