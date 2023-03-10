#!/usr/bin/env python
"""Randomise the selected item's colours using hsl colorspace"""

from random import randrange, uniform
import inkex

def _rand(limit, value, roof=255, method=randrange, circular=False):
    """Generates a random number which is less than limit % away from value, using the method
    supplied."""
    limit = roof * float(limit) / 100
    limit /= 2
    max_ = type(roof)(value + limit)
    min_ = type(roof)(value - limit)
    if not (circular):
        if max_ > roof:
            min_ -= max_ - roof
            max_ = roof
        if min_ < 0:
            max_ -= min_
            min_ = 0
        return method(min_, max_)
    else:
        return method(min_, max_) % roof


class Randomize(inkex.ColorExtension):
    """Randomize the colours of all objects"""
    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument("-y", "--hue_range", type=int, default=0, help="Hue range")
        pars.add_argument("-t", "--saturation_range", type=int, default=0, help="Saturation range")
        pars.add_argument("-m", "--lightness_range", type=int, default=0, help="Lightness range")
        pars.add_argument("-o", "--opacity_range", type=int, default=0, help="Opacity range")


    def modify_color(self, name, color):
        hsl = color.to_hsl()
        if self.options.hue_range > 0:
            hsl.hue = int(_rand(self.options.hue_range, hsl.hue, circular=True))
        if self.options.saturation_range > 0:
            hsl.saturation = int(_rand(self.options.saturation_range, hsl.saturation))
        if self.options.lightness_range > 0:
            hsl.lightness = int(_rand(self.options.lightness_range, hsl.lightness))
        return hsl.to_rgb()

    def modify_opacity(self, name, opacity):
        try:
            opacity = float(opacity)
        except ValueError:
            self.msg(f"Ignoring unusual opacity value: {opacity}")
            return opacity
        orange = self.options.opacity_range
        if orange > 0:
            return _rand(orange, opacity, roof=1.0, method=uniform)
        return opacity

if __name__ == '__main__':
    Randomize().run()
