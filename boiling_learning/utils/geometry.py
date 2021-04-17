import numpy as np
import pint
from dataclassy import dataclass


@dataclass(frozen=True)
class Solid:
    surface_area: pint.Quantity = None
    volume: pint.Quantity = None


@dataclass(frozen=True)
class Prism(Solid):
    length: pint.Quantity

    cross_section_perimeter: pint.Quantity = None
    cross_section_area: pint.Quantity = None

    lateral_area: pint.Quantity = None
    surface_area: pint.Quantity = None
    volume: pint.Quantity = None

    def __init__(self) -> None:
        self.lateral_area = self.cross_section_perimeter * self.length
        self.surface_area = self.lateral_area + 2*self.cross_section_area
        self.volume = self.cross_section_area * self.length


@dataclass(frozen=True)
class Cylinder(Prism):
    diameter: pint.Quantity

    def __init__(self) -> None:
        self.radius = self.diameter / 2
        self.cross_section_perimeter = np.pi * self.diameter
        self.cross_section_area = np.pi * self.radius**2

        super().__init__()


@dataclass(frozen=True)
class RectangularPrism(Prism):
    width: pint.Quantity
    thickness: pint.Quantity

    def __init__(self) -> None:
        self.cross_section_perimeter = 2*(self.width + self.thickness)
        self.cross_section_area = self.width * self.thickness

        super().__init__()
