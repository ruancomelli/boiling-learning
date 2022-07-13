import numpy as np

from boiling_learning.io.storage import load, save
from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.pathutils import tempdir


def test_save_frame() -> None:
    frame: VideoFrame = np.random.random((10, 20, 3))

    with tempdir() as temp:
        path = temp / 'frame'

        save(frame, path)
        other_frame = load(path)

    assert np.isclose(frame, other_frame, 1e-8).all()
