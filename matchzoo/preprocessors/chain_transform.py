"""Wrapper function organizes a number of transform functions."""
import typing

from .units.unit import Unit


class ChainTransform(object):
    """
    Compose unit transformations into a single function.
    """

    def __init__(self, units: typing.List[Unit]):
        """
        :param units: List of :class:`matchzoo.StatelessUnit`.
        """
        self.units = units

    def __call__(self, text):
        for unit in self.units:
            text = unit.transform(text)
        return text

    def __str__(self):
        unit_names = ' => '.join(unit.__class__.__name__
                                 for unit in self.units)
        return 'Chain Transform of ' + unit_names

    @property
    def __name__(self):
        return self.__str__()
