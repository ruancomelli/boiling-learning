import contextlib
import typing
from fractions import Fraction
from numbers import Real
from typing import Any, Dict, Iterable, Tuple, Union

import imageio
import numpy as np
from PIL import Image, ImageDraw

from boiling_learning.preprocessing.video import VideoFrame
from boiling_learning.utils.pathutils import PathLike, resolve


def save_as_video(
    path: PathLike,
    frames: Iterable[Tuple[VideoFrame, Dict[str, Any]]],
    *,
    display_data: Union[str, Tuple[str, ...], Dict[str, str]] = (),
    text_position: Union[Tuple[int, int], Tuple[Fraction, Fraction]] = (
        Fraction(1, 10),
        Fraction(1, 10),
    ),
    text_color: int = 255,
    fps: Real,
    fmt: str = 'mp4',
) -> None:
    path = resolve(path, parents=True)
    if isinstance(display_data, str):
        display_data = (display_data,)

    with contextlib.closing(
        imageio.get_writer(str(path), format=fmt, mode='I', fps=float(fps))
    ) as writer:
        for frame, data in frames:
            image = Image.fromarray((frame.squeeze() * 255).astype(np.uint8))
            annotated_image = _annotate_image(
                image,
                data,
                display_data=display_data,
                text_position=text_position,
                text_color=text_color,
            )
            writer.append_data(np.array(annotated_image))


def save_as_gif(
    path: PathLike,
    frames: Iterable[Tuple[VideoFrame, Dict[str, Any]]],
    *,
    display_data: Union[str, Tuple[str, ...], Dict[str, str]] = (),
    text_position: Union[Tuple[int, int], Tuple[Fraction, Fraction]] = (
        Fraction(1, 10),
        Fraction(1, 10),
    ),
    text_color: int = 255,
    duration: int,
) -> None:
    path = resolve(path, parents=True)
    if isinstance(display_data, str):
        display_data = (display_data,)

    imgs = (
        _annotate_image(
            Image.fromarray(frame.squeeze() * 255),
            data,
            display_data=display_data,
            text_position=text_position,
            text_color=text_color,
        )
        for frame, data in frames
    )

    img = next(imgs)
    img.save(str(path), format='GIF', append_images=imgs, save_all=True, duration=duration)


def _annotate_image(
    image: Image.Image,
    data: Dict[str, Any],
    *,
    display_data: Union[Tuple[str, ...], Dict[str, str]],
    text_position: Union[Tuple[int, int], Tuple[Fraction, Fraction]],
    text_color: int = 255,
) -> Image.Image:
    absolute_text_position = _to_absolute_position(image, text_position)

    display_items = (
        (f'{key}: {data[key]:.0f}' for key in display_data)
        if isinstance(display_data, tuple)
        else (
            f'{translated}: {data[key]:.0f}' if translated else f'{data[key]:.0f}'
            for key, translated in display_data.items()
        )
    )
    text = '\n'.join(display_items)

    if text:
        draw = ImageDraw.Draw(image)
        draw.text(absolute_text_position, text, fill=text_color)

    return image


def _to_absolute_position(
    image: Image.Image, position: Union[Tuple[int, int], Tuple[Fraction, Fraction]]
) -> Tuple[int, int]:
    x_position, y_position = position
    if isinstance(x_position, int):
        return typing.cast(Tuple[int, int], position)

    return (round(x_position * image.width), round(y_position * image.height))
