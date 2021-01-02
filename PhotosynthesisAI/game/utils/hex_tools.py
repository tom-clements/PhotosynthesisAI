from typing import Tuple

import numpy as np
import hexy as hx

from functools import lru_cache

from .constants import BOARD_RADIUS
from .utils import time_function


@lru_cache(maxsize=None)
@time_function
def get_coord_radius(coords: Tuple[int]) -> int:
    return int(sum(abs(np.array(coords))) / 2)


@lru_cache(maxsize=None)
@time_function
def get_surrounding_coords(coords: Tuple[int], radius: int, max_radius: int) -> Tuple:
    surrounding_tile_coords = hx.get_disk(coords, radius)
    surrounding_tile_coords = [
        tuple(coord)
        for coord in surrounding_tile_coords
        if (max(np.absolute(coord)) <= max_radius) & (not all(coord == coords))
    ]
    return tuple(surrounding_tile_coords)


@lru_cache(maxsize=None)
@time_function
def get_coords_at_sun_edge(sun: Tuple[int], board_coords: Tuple) -> Tuple:
    board_coords = np.array([np.array(coord) for coord in board_coords])
    mask = (
        np.count_nonzero((np.array([np.array(sun) for t in board_coords]) == board_coords / 3) * np.array(sun), axis=1)
        > 0
    )
    edge_coords = board_coords[mask]
    return edge_coords


@lru_cache(maxsize=None)
@time_function
def get_coords_along_same_axis(coords: Tuple[int], axis: Tuple[int], max_radius: int) -> Tuple[Tuple[int]]:
    tiles = []
    coords = np.array(coords)
    while max(abs(coords)) <= max_radius:
        tiles.append(tuple(coords))
        coords = coords - axis
    return tuple(tiles)


@lru_cache(maxsize=None)
@time_function
def notation_from_coords(coords: Tuple[int, int, int]) -> str:
    letter_number = int(coords[0]) + BOARD_RADIUS + 1
    letter = chr(ord("`") + letter_number)
    rank_start_number = -int(coords[0]) - BOARD_RADIUS if coords[0] <= 0 else -3
    number = int(coords[1]) - rank_start_number + 1
    return f"{letter}{int(number)}"
