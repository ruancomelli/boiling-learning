import utils

filepath = project_path / 'cases' / 'case 1' / 'videos' / 'GOPR2850.MP4'
outputdir = project_path / 'testing_extract'

utils.rmdir(outputdir, recursive=True, keep=True)

def chunkify(x, chunksize):
    return ((x // chunksize) * chunksize, (x // chunksize + 1) * chunksize - 1)

def final_pattern(index):
    return Path('from_{}_to_{}'.format(*chunkify(index, 10))) / f'my_frame_{index}.png'

frame_list = (
    (frame_number, final_pattern(frame_number))
    for frame_number in range(200)
)

# frame_list = (
#     (frame_number, f'frame_{frame_number}')
#     for frame_number in range(200)
# )

utils.video.extract_frames(
    filepath,
    outputdir,
    # frame_list=frame_list,
    final_pattern=final_pattern,
    # final_pattern='my_frame_{index}.png',
    verbose=True
)