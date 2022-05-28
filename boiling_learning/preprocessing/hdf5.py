import typing
from typing import Iterator, Tuple

import h5py
import hdf5plugin
import more_itertools as mit
import numpy as np
from loguru import logger

from boiling_learning.preprocessing.transformers import Transformer
from boiling_learning.preprocessing.video import Video, VideoFrame, open_video
from boiling_learning.utils.utils import PathLike, resolve


def video_to_hdf5(
    video: Video,
    dest: PathLike,
    *,
    dataset_name: str,
    batch_size: int = 1,
    transformers: Tuple[Transformer[VideoFrame, VideoFrame], ...] = (),
) -> None:
    """Save video as an HDF5 file

    Arguments:
        transformers: a tuple of transformers mapping a batch of frames to
        another batch of frames. Suitable for broadcasting image transformers.
    """
    destination = str(resolve(dest))

    # use OpenCV reader because it is much more memory efficient than PIMS
    frames = video_frames(video.path)
    frame_chunks = mit.chunked(frames, batch_size)
    array_chunks = map(np.array, frame_chunks)

    with h5py.File(destination, 'a') as file:
        dataset = file.require_dataset(
            dataset_name,
            video.shape,
            dtype='f',
            # best compression algorithm I found - good compression, fastest decompression
            **hdf5plugin.LZ4(),
        )
        for index, batch in enumerate(array_chunks):
            start, end = index * batch_size, (index + 1) * batch_size

            logger.debug(f'Writing frames {start}:{end} from {video.path} to {destination}')

            for transformer in transformers:
                batch = map(transformer, batch)

            batch = np.array(batch)

            dataset.write_direct(batch, dest_sel=np.s_[start:end])
            dataset.flush()
            file.flush()


def get_frame_from_hdf5(path: PathLike, index: int, *, dataset_name: str) -> VideoFrame:
    resolved = str(resolve(path))

    with h5py.File(resolved, 'r', swmr=True) as f:
        return typing.cast(VideoFrame, f[dataset_name][index])


def video_frames(video_path: PathLike) -> Iterator[VideoFrame]:
    with open_video(video_path) as cap:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame
