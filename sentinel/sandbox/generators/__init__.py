"""Synthetic market generators.

Two of these contain no directional signal at all and serve as controls:
`GBMGenerator` (the clean null) and `JumpDiffusionGenerator` (a null with fat
tails). `HestonGenerator` has no directional signal either, but does have
forecastable volatility. The rest contain structure of a known kind and strength.
"""

from sentinel.sandbox.generators.ar1 import AR1Generator
from sentinel.sandbox.generators.base import Generator, Simulation
from sentinel.sandbox.generators.gbm import GBMGenerator
from sentinel.sandbox.generators.heston import HestonGenerator
from sentinel.sandbox.generators.jump import JumpDiffusionGenerator
from sentinel.sandbox.generators.ou import OUGenerator
from sentinel.sandbox.generators.regime import RegimeSwitchingGenerator

__all__ = [
    "Generator",
    "Simulation",
    "GBMGenerator",
    "AR1Generator",
    "OUGenerator",
    "RegimeSwitchingGenerator",
    "JumpDiffusionGenerator",
    "HestonGenerator",
]
