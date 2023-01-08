from pathlib import Path

import tensorflow as tf
import typer

from boiling_learning.app.constants import BOILING_BASELINE_BATCH_SIZE
from boiling_learning.app.datasets.bridged.boiling1d import DEFAULT_BOILING_OUTLIER_FILTER
from boiling_learning.app.datasets.bridging import to_tensorflow_triplet
from boiling_learning.app.datasets.preprocessed.boiling1d import boiling_datasets
from boiling_learning.app.paths import studies_path
from boiling_learning.datasets.splits import DatasetTriplet
from boiling_learning.image_datasets import ImageDatasetTriplet
from boiling_learning.lazy import LazyDescribed
from boiling_learning.visualization.video import save_as_video

app = typer.Typer()


@app.command()
def boiling1d(
    direct: bool = typer.Option(..., '--direct/--indirect'),
    tensorflow: bool = typer.Option(False),
    each: int = typer.Option(60),
    fps: int = typer.Option(30),
) -> None:
    datasets = boiling_datasets(direct_visualization=direct)

    paths = {
        (dataset, subset_name): path
        for index, dataset in enumerate(datasets)
        for subset_name in ('train', 'val', 'test')
        if not (
            path := _animations_path()
            / (
                'boiling-case'
                f'-{index + 1}'
                f"-{'direct' if direct else 'indirect'}"
                f'-{subset_name}'
                f'-each-{each}'
                f"{'-tf' if tensorflow else ''}"
                '.mp4'
            )
        ).is_file()
    }

    for (dataset, subset_name), path in paths.items():
        triplet = (_tensorflow_datasets if tensorflow else _sliceable_datasets)(
            dataset,
            each,
        )
        subset_index = {'train': 0, 'val': 1, 'test': 2}[subset_name]
        subset = triplet[subset_index]

        save_as_video(
            path,
            subset,
            display_data={'index': 'Index', 'Flux [W/cm**2]': 'Flux [W/cm²]'},
            fps=fps,
        )


def _tensorflow_datasets(
    dataset: LazyDescribed[ImageDatasetTriplet],
    each: int,
) -> DatasetTriplet[tf.data.Dataset]:
    return tuple(
        subset()
        .unbatch()
        .enumerate()
        .filter(lambda count, item: count % each == 0)
        .map(lambda count, item: item)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
        for subset in to_tensorflow_triplet(
            dataset,
            prefilterer=DEFAULT_BOILING_OUTLIER_FILTER,
            batch_size=BOILING_BASELINE_BATCH_SIZE,
            experiment='boiling1d',
            shuffle=False,
        )
    )


def _sliceable_datasets(
    dataset: LazyDescribed[ImageDatasetTriplet],
    each: int,
) -> ImageDatasetTriplet:
    return tuple(subset[::each].prefetch(1024) for subset in dataset())


@app.command()
def condensation(
    fps: int = typer.Option(30),
) -> None:
    raise NotImplementedError


def _animations_path() -> Path:
    return studies_path() / 'animations'
