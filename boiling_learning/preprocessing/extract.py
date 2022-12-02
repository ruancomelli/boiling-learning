import ffmpeg
from loguru import logger

from boiling_learning.preprocessing.video import Video
from boiling_learning.utils.pathutils import PathLike, resolve


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    filename_pattern: str | None = None,
) -> None:
    # Original code: $ ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"
    # Source 2: <https://forums.fast.ai/t/extracting-frames-from-video-file-with-ffmpeg/29818>

    video_path = resolve(video_path)
    output_dir = resolve(output_dir, dir=True)

    logger.info(f'Extracting frames from {video_path} to {output_dir}')

    if filename_pattern is None:
        number_of_frames = len(Video(video_path))
        number_of_digits = len(str(number_of_frames))

        filename_pattern = f'%{number_of_digits}d.png'

    (
        ffmpeg.input(str(video_path))
        .output(str(output_dir / filename_pattern), format='image2')
        .run()
    )
