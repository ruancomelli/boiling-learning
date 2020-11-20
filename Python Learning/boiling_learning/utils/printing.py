import pint


def compressed_unit(unit: pint.Unit) -> str:
    return f'{unit:~}'.replace(' ', '')


def unit_post_fix(unit: pint.Unit) -> str:
    return '[' + compressed_unit(unit) + ']'


def add_unit_post_fix(text: str, unit: pint.Unit) -> str:
    return ' '.join((text, unit_post_fix(unit)))
