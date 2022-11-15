import abc
import math

from pint import Quantity


class Prism(abc.ABC):
    @abc.abstractmethod
    def length(self) -> Quantity:
        pass

    @abc.abstractmethod
    def cross_section_perimeter(self) -> Quantity:
        pass

    @abc.abstractmethod
    def cross_section_area(self) -> Quantity:
        pass

    def lateral_area(self) -> Quantity:
        return self.cross_section_perimeter() * self.length()

    def surface_area(self) -> Quantity:
        return self.lateral_area() + 2 * self.cross_section_area()

    def volume(self) -> Quantity:
        return self.cross_section_area() * self.length()


class Cylinder(Prism):
    def __init__(self, length: Quantity, diameter: Quantity) -> None:
        self._length = length
        self._diameter = diameter

    def length(self) -> Quantity:
        return self._length

    def diameter(self) -> Quantity:
        return self._diameter

    def radius(self) -> Quantity:
        return self._diameter / 2

    def cross_section_perimeter(self) -> Quantity:
        return math.pi * self.diameter()

    def cross_section_area(self) -> Quantity:
        return math.pi * self.radius() ** 2


class RectangularPrism(Prism):
    def __init__(self, length: Quantity, width: Quantity, thickness: Quantity) -> None:
        self._length = length
        self._width = width
        self._thickness = thickness

    def length(self) -> Quantity:
        return self._length

    def width(self) -> Quantity:
        return self._width

    def thickness(self) -> Quantity:
        return self._thickness

    def cross_section_perimeter(self) -> Quantity:
        return 2 * (self.width() + self.thickness())

    def cross_section_area(self) -> Quantity:
        return self.width() * self.thickness()
