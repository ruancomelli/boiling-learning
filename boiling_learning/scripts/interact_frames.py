# import warnings
# from typing import Any, Callable, Iterable

# import funcy
# import numpy as np
# from ipywidgets import interact, widgets
# from skimage.transform import downscale_local_mean as downscale

# from boiling_learning.preprocessing.image_datasets import ExperimentVideoDataset
# from boiling_learning.preprocessing.experiment_video import ExperimentVideo

# try:
#     from google.colab.patches import cv2_imshow as google_colab_imshow
# except ImportError:
#     google_colab_imshow = None

# try:
#     from cv2 import cv_imshow
# except ImportError:
#     cv_imshow = None


# def _make_show_frame_function(
#     imshow: Callable[[np.ndarray], Any],
#     downscale_factor: Iterable[int] = (8, 8, 1),
# ) -> Callable[[np.ndarray], None]:
#     def _imshow(image: np.ndarray) -> None:
#         # since we are just visualizing, there is no need to retain
#         # the full image resolution
#         image = downscale(image, downscale_factor)
#         imshow(image)

#     return _imshow


# def _interact_dataset_frames(
#     datasets: tuple[ExperimentVideoDataset, ...], imshow: Callable[[np.ndarray], Any]
# ) -> None:
#     datasets_options = [(str(dataset), dataset) for dataset in datasets]
#     default_dataset = datasets_options[0][1]
#     datasets_widget = widgets.Dropdown(
#         options=datasets_options,
#         description='Dataset name:',
#         value=default_dataset,
#     )

#     experiment_videos_options = [
#         (experiment_video.name, experiment_video)
#         for experiment_video in datasets_widget.value.values()
#     ]
#     experiment_videos_widget = widgets.Dropdown(
#         options=experiment_videos_options,
#         description='Experiment video:',
#         value=experiment_videos_options[0][1],
#     )

#     def update_videos_list(changes):
#         experiment_videos_widget.options = [
#             (experiment_video.name, experiment_video)
#             for experiment_video in changes['new'].values()
#         ]

#     datasets_widget.observe(update_videos_list, 'value')

#     index_widget = widgets.IntSlider(
#         value=0,
#         min=0,
#         max=len(experiment_videos_widget.value) - 1,
#         description='Frame:',
#     )

#     def update_max_index(changes):
#         index_widget.max = len(changes['new']) - 1

#     experiment_videos_widget.observe(update_max_index, 'value')

#     def show_frames(dataset: ExperimentVideoDataset, ev: ExperimentVideo, idx: int) -> None:
#         imshow(ev[idx])

#     interact(
#         show_frames,
#         dataset=datasets_widget,
#         ev=experiment_videos_widget,
#         idx=index_widget,
#     )


# def main(datasets: Iterable[ExperimentVideoDataset], colab_backend: bool = False) -> None:
#     datasets = tuple(datasets)

#     imshow = (
#         google_colab_imshow if colab_backend and google_colab_imshow is not None else cv_imshow
#     )

#     # rescale images since float images are considered to be in [0, 1]
#     # but cv2_imshow expects them to be [0, 255]
#     imshow = funcy.compose(imshow, lambda img: img * 255)
#     imshow = _make_show_frame_function(imshow)

#     with warnings.catch_warnings():
#         _interact_dataset_frames(datasets, imshow)
