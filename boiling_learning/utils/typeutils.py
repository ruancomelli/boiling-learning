import typeguard


def isinstance_(obj, type_) -> bool:
    try:
        typeguard.check_type('', obj, type_)
        return True
    except TypeError:
        return False
