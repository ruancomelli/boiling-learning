import warnings
from typing import Any, Callable, Iterable, Tuple

import funcy
import numpy as np
from ipywidgets import interact, widgets
from skimage.transform import downscale_local_mean as downscale

from boiling_learning.preprocessing import ImageDataset
from boiling_learning.preprocessing.ExperimentVideo import ExperimentVideo


def _make_show_frame_function(
    imshow: Callable[[np.ndarray], Any],
    downscale_factor: Iterable[int] = (8, 8, 1),
) -> Callable[[np.ndarray], None]:
    def _imshow(image: np.ndarray) -> None:
        # since we are just visualizing, there is no need to retain
        # the full image resolution
        image = downscale(image, downscale_factor)
        imshow(image)

    return _imshow


def _interact_dataset_frames(
    datasets: Tuple[ImageDataset, ...], imshow: Callable[[np.ndarray], Any]
) -> None:
    datasets_options = [(dataset.name, dataset) for dataset in datasets]
    default_dataset = datasets_options[0][1]
    datasets_widget = widgets.Dropdown(
        options=datasets_options,
        description='Dataset name:',
        value=default_dataset,
    )

    experiment_videos_options = [
        (experiment_video.name, experiment_video)
        for experiment_video in datasets_widget.value.values()
    ]
    experiment_videos_widget = widgets.Dropdown(
        options=experiment_videos_options,
        description='Experiment video:',
        value=experiment_videos_options[0][1],
    )

    def update_videos_list(changes):
        experiment_videos_widget.options = [
            (experiment_video.name, experiment_video)
            for experiment_video in changes['new'].values()
        ]

    datasets_widget.observe(update_videos_list, 'value')

    index_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=len(experiment_videos_widget.value) - 1,
        description='Frame:',
    )

    def update_max_index(changes):
        index_widget.max = len(changes['new']) - 1

    experiment_videos_widget.observe(update_max_index, 'value')

    def show_frames(dataset: ImageDataset, ev: ExperimentVideo, idx: int):
        imshow(ev[idx])

    interact(
        show_frames,
        dataset=datasets_widget,
        ev=experiment_videos_widget,
        idx=index_widget,
    )


def main(
    datasets: Iterable[ImageDataset], colab_backend: bool = False
) -> None:
    datasets = tuple(datasets)

    imshow_imported: bool = False
    if colab_backend:
        try:
            from google.colab.patches import cv2_imshow as imshow

            imshow_imported = True
        except (ImportError, ModuleNotFoundError):
            pass

    if not imshow_imported:
        from cv2 import imshow

    # rescale images since float images are considered to be in [0, 1]
    # but cv2_imshow expects them to be [0, 255]
    imshow = funcy.compose(imshow, lambda img: img * 255)
    imshow = _make_show_frame_function(imshow)

    with warnings.catch_warnings():
        _interact_dataset_frames(datasets, imshow)


if __name__ == '__main__':
    raise RuntimeError(
        '*interact_frames* cannot be executed as a standalone script yet.'
    )
