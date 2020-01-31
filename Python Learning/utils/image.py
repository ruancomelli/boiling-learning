from utils import utils

def shift(
    image,
    shifts=None,
    shift_x=None,
    shift_y=None
):
    idx = utils.check_value_match(
        [
            {
                'shifts': lambda x: x is None,
                'shift_x': lambda x: x is not None,
                'shift_y': lambda x: x is not None,
            },
            {
                'shifts': lambda x: x is not None,
                'shift_x': lambda x: x is None,
                'shift_y': lambda x: x is None,
            }
        ],
        {
            'shifts': shifts,
            'shift_x': shift_x,
            'shift_y': shift_y
        }
    )
    if idx == 0:
        return shift(image, shifts=(shift_x, shift_y))
    
    def _shift(image, shifts, axis=0):
        from numpy import roll
        
        if shifts:
            return _shift(
                roll(image, shifts[0], axis=axis),
                shifts[1:],
                axis+1
            )
        else:
            return image
    return _shift(image, shifts, axis=0)

def crop(
    image,
    lims=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    x_lim=None,
    y_lim=None
):
    idx = utils.check_value_match(
        [
            dict(
                lims=lambda x: x is not None,
                x_min=lambda x: x is None,
                x_max=lambda x: x is None,
                y_min=lambda x: x is None,
                y_max=lambda x: x is None,
                x_lim=lambda x: x is None,
                y_lim=lambda x: x is None
            ),
            dict(
                lims=lambda x: x is None,
                x_min=lambda x: x is not None,
                x_max=lambda x: x is not None,
                y_min=lambda x: x is not None,
                y_max=lambda x: x is not None,
                x_lim=lambda x: x is None,
                y_lim=lambda x: x is None
            ),
            dict(
                lims=lambda x: x is None,
                x_min=lambda x: x is None,
                x_max=lambda x: x is None,
                y_min=lambda x: x is None,
                y_max=lambda x: x is None,
                x_lim=lambda x: x is not None,
                y_lim=lambda x: x is not None
            )
        ],
        dict(
            lims=lims,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_lim=x_lim,
            y_lim=y_lim
        )
    )
    if idx == 0:
        return image[tuple(slice(*lim) for lim in lims)]
    elif idx == 1:
        return crop(image, lims=[(x_min, x_max), (y_min, y_max)])
    elif idx == 2:
        return crop(image, lims=[x_lim, y_lim])