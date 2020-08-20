import hexy as hx
import numpy as np

class Tile:
    def __init__(self, tree, coords, index):
        self.tree = tree
        self.coords = coords
        self.index = index
        self.is_locked = False
        self.is_shadow = False
        
    def get_adjacent_tile(self, axis) -> np.ndarray:
        return self.coords + axis
    
    def get_surrounding_coords(self, radius: int) -> np.ndarray:
        surrounding_tile_coords = hx.get_disk(self.coords, radius)
        return surrounding_tile_coords