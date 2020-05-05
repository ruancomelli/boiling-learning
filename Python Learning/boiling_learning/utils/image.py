from skimage.color import rgb2gray as grayscale
from skimage.io import imread, imsave
from skimage.transform import AffineTransform, downscale_local_mean, warp
from skimage.util import crop as skimage_crop

# TODO: can wrappers remove code repetition in crop and shift for the "if image is None: image = imread(in_path)" and the "if out_path is not None: imsave(out_path)"?

def crop(
        image=None,
        in_path=None,
        out_path=None,
        crop_dict=None,
        crop_tuple=None,
        top=0,
        bottom=0,
        left=0,
        right=0
    ):
    # source: <https://stackoverflow.com/questions/33287613/crop-image-in-skimage>
    # TODO: check_value_match here
        
    if crop_tuple is None:        
        if crop_dict is None:
            crop_tuple = (
                (top, bottom),
                (left, right)
            )
        else:
            crop_tuple = (
                (crop_dict.get('top', top), crop_dict.get('bottom', bottom)),
                (crop_dict.get('left', left), crop_dict.get('right', right))
            )
        if image.ndim == 3:
            # compensate in case image is 3D (RGB)
            crop_tuple += ((0, 0),)
        return crop(image, in_path, out_path, crop_tuple=crop_tuple)

    if image is None:
        image = imread(in_path)
        
    cropped = skimage_crop(image, crop_tuple)
    if out_path is not None:
        imsave(out_path, cropped)
        
    return cropped
    
def shift(
        image=None,
        in_path=None, out_path=None,
        shifts=None,
        shift_left=None,
        shift_right=None,
        shift_up=None,
        shift_down=None
):
    # source: <https://stackoverflow.com/questions/47961447/shift-image-in-scikit-image-python>
    # TODO: check_value_match here!
    
    if shifts is None:
        if shift_left is None:
            shift_left = - shift_right
        if shift_up is None:
            shift_up = - shift_down
        shifts = (shift_left, shift_up)
    
    if image is None:
        image = imread(in_path)
        
    transform = AffineTransform(translation=shifts)
    shifted = warp(image, transform, mode='wrap', preserve_range=True)
    shifted = shifted.astype(image.dtype)
    
    if out_path is not None:
        imsave(out_path, shifted)
        
    return shifted

def flip(image, horizontal=False, vertical=False):
    # TODO: check_value_match here!
    # use in_path and out_path?
    
    if horizontal and not vertical:
        return image[:, ::-1, ...]
    elif vertical and not horizontal:
        return image[::-1, ...]
    elif horizontal and vertical:
        return image[::-1, ::-1, ...]
    else:
        return image
    
def downscale(image, shape): 
    if isinstance(shape, int):
        if image.ndim == 2:
            shape = (shape, shape)
        else:
            shape = (shape, shape, 1)
    elif isinstance(shape, tuple):
        shape = shape + (1,)*(img.ndim - len(shape))
    
    return downscale_local_mean(image, shape)
    
# def grayscale(image, **kwargs): 
#     return rgb2gray(image, **kwargs)

# greyscale = grayscale