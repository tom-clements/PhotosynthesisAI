import hexy as hx
import numpy as np
from .tree import Tree
from PhotosynthesisAI.game.utils.constants import RICHNESS


class Tile:
    def __init__(self, tree: Tree, coords: np.ndarray, index: int, is_locked: bool = False, is_shadow: bool = False):
        self.tree = tree
        self.coords = coords
        self.index = index
        self.is_locked = is_locked
        self.is_shadow = is_shadow
        self.richness = self._get_richness()

    def get_adjacent_tile(self, axis: np.ndarray) -> np.ndarray:
        return self.coords + axis

    def get_surrounding_coords(self, radius: int) -> np.ndarray:
        surrounding_tile_coords = hx.get_disk(self.coords, radius)
        return surrounding_tile_coords

    def _get_richness(self) -> int:
        radius = int(sum(abs(self.coords))/2)
        return RICHNESS[radius]
