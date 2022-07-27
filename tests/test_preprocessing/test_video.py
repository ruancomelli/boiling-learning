from pathlib import Path

import numpy as np

from boiling_learning.io.storage import load, save
from boiling_learning.preprocessing.video import VideoFrame


def test_save_frame(tmp_path: Path) -> None:
    frame: VideoFrame = np.random.random((10, 20, 3))

    path = tmp_path / 'frame'
    save(frame, path)
    other_frame = load(path)

    assert np.isclose(frame, other_frame, 1e-8).all()
