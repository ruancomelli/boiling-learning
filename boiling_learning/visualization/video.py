import contextlib
import typing
from fractions import Fraction
from numbers import Real
from typing import Dict, Iterable, Tuple, Union

import imageio
import numpy as np
from PIL import Image, ImageDraw

from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.pathutils import PathLike, resolve


def save_as_video(
    path: PathLike,
    frames: Iterable[VideoFrame],
    display_data: Union[Tuple[str, ...], Dict[str, str]],
    *,
    fps: Real,
    fmt: str = 'mp4',
    text_position: Union[Tuple[int, int], Tuple[Fraction, Fraction]],
    text_color: int = 255,
) -> None:
    path = resolve(path, parents=True)

    with contextlib.closing(
        imageio.get_writer(str(path), format=fmt, mode='I', fps=float(fps))
    ) as writer:
        for frame, data in frames:
            image = Image.fromarray((frame.squeeze() * 255).astype(np.uint8))
            absolute_text_position = _to_absolute_position(frame, text_position)

            display_items = (
                (f'{key}: {data[key]:.0f}' for key in display_data)
                if isinstance(display_data, tuple)
                else (
                    f'{translated}: {data[key]:.0f}' if translated else f'{data[key]:.0f}'
                    for key, translated in display_data.items()
                )
            )
            text = '\n'.join(display_items)

            draw = ImageDraw.Draw(image)
            draw.text(absolute_text_position, text, fill=text_color)

            writer.append_data(np.array(image))


def _to_absolute_position(
    frame: VideoFrame, position: Union[Tuple[int, int], Tuple[Fraction, Fraction]]
) -> Tuple[int, int]:
    x_position, y_position = position
    if isinstance(x_position, int):
        return typing.cast(Tuple[int, int], position)

    height, width, _ = frame.shape
    return (round(x_position * width), round(y_position * height))
