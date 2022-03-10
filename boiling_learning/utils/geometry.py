import math

from dataclassy import dataclass
from pint import Quantity


@dataclass(kwargs=True)
class Solid:
    surface_area: Quantity = None
    volume: Quantity = None


class Prism(Solid):
    length: Quantity

    cross_section_perimeter: Quantity
    cross_section_area: Quantity

    lateral_area: Quantity = None

    def __post_init__(self) -> None:
        self.lateral_area = self.cross_section_perimeter * self.length
        self.surface_area = self.lateral_area + 2 * self.cross_section_area


class Cylinder(Prism):
    diameter: Quantity

    cross_section_perimeter: Quantity = None
    cross_section_area: Quantity = None

    def __post_init__(self) -> None:
        self.radius = self.diameter / 2
        self.cross_section_perimeter = math.pi * self.diameter
        self.cross_section_area = math.pi * self.radius**2
        super().__post_init__()


class RectangularPrism(Prism):
    width: Quantity
    thickness: Quantity

    cross_section_perimeter: Quantity = None
    cross_section_area: Quantity = None

    def __post_init__(self) -> None:
        self.cross_section_perimeter = 2 * (self.width + self.thickness)
        self.cross_section_area = self.width * self.thickness
