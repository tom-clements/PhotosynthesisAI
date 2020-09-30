from typing import Union

import hexy as hx
import numpy as np

from .tree import Tree
from PhotosynthesisAI.game.utils.constants import RICHNESS
from ..utils.hex_tools import _get_coord_radius
from ..utils.utils import time_function


class Tile:
    def __init__(
        self, tree: Union[Tree, None], coords: np.ndarray, index: int, is_locked: bool = False, is_shadow: bool = False
    ):
        self.tree = tree
        self.coords = coords
        self.index = index
        self.is_locked = is_locked
        self.is_shadow = is_shadow
        self.richness = self._get_richness()

    # def get_adjacent_tile(self, axis: np.ndarray) -> np.ndarray:
    #     return self.coords + axis

    @time_function
    def _get_richness(self) -> int:
        radius = _get_coord_radius(tuple(self.coords))
        return RICHNESS[radius]
