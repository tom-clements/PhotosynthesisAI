from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

from PhotosynthesisAI.game.components import Tree, Tile
from PhotosynthesisAI.game.utils.constants import PLANT_LP_COST, COLLECT_LP_COST, TREES, BOARD_RADIUS
from PhotosynthesisAI.game.utils.utils import time_function


class Move:
    def __init__(self, board: "Board", tile: Tile = None, from_tile: Tile = None, tree: Tree = None, cost: int = None):
        self.board = board
        self.tile = tile
        self.from_tile = from_tile
        self.tree = tree
        self.cost = cost
        self._set_move_id()

    # this function gets hit a lot - contributes to 6% of compute
    @time_function
    def _set_move_id(self):
        num_tile_actions = len(self.board.data.tiles)
        move_name = self.__class__.__name__
        tile_index = self.tile.index if self.tile else None
        from_tile_index = self.from_tile.index if self.from_tile else None
        tree_size = self.tree.size if self.tree else None
        self.move_id = self._get_move_index(move_name, num_tile_actions, tile_index, from_tile_index, tree_size)

    @lru_cache(maxsize=None)
    def _get_plant_move_index(self, from_tile_index, to_tile_index):
        if from_tile_index:
            # [[(from_tile, to_tile), (from_tile1, to_tile2), ...], [(from_tile2, to_tile_x) ...]]
            surrounding_tiles = [
                [
                    (self.board.get_tile_index(tuple(from_coords)), tile.index)
                    for tile in self.board.get_surrounding_tiles(tuple(from_coords), radius=3)
                ]
                for from_coords in self.board.tile_coords
            ]
            # flatten
            # [(from_tile, to_tile), (from_tile1, to_tile2), ..., (from_tile2, to_tile_x) ...]
            surrounding_tiles = [item for sublist in surrounding_tiles for item in sublist]
            i = 0
            for tiles in surrounding_tiles:
                if (from_tile_index == tiles[0]) and (to_tile_index == tiles[1]):
                    return i
                i += 1
        else:
            # this is for starting moves
            num_edge_tiles = (BOARD_RADIUS + 1) * 6
            edge_tiles = [self.board.get_tile_index(tuple(coords)) for coords in self.board.tile_coords if max(abs(coords)) == 3]
            i = 0
            for tile in edge_tiles:
                if to_tile_index == tile:
                    break
                i += 1

            if i == len(edge_tiles):
                raise ValueError("Out of Bounds")
            return self._number_of_non_starting_plant_moves() + i

        raise ValueError("Out of bounds")

    def _number_of_non_starting_plant_moves(self):
        # each plant comes from a tree, so each tile has a radius 3 area around of possible plant moves
        return sum(
            [len(self.board.get_surrounding_tiles(tuple(coords), radius=3)) for coords in self.board.tile_coords]
        )


    @lru_cache(maxsize=None)
    @time_function
    def _get_move_index(
        self, move_name: str, num_tile_actions: int, tile_index: int, from_tile_index: int, tree_size: int
    ) -> int:
        num_buy_moves = len(TREES.keys())
        starting_index = {
            "Grow": 0,
            "Collect": num_tile_actions * 1,
            "Buy": num_tile_actions * 2,
            "EndGo": num_tile_actions * 2 + num_buy_moves,
            "Plant": num_tile_actions * 2 + num_buy_moves + 1,
        }
        if move_name in ["Grow", "Collect"]:
            move_index = starting_index[move_name] + tile_index
        elif move_name == "Buy":
            move_index = starting_index[move_name] + tree_size
        elif move_name == "Plant":
            plant_move_index = self._get_plant_move_index(from_tile_index, tile_index)
            move_index = starting_index[move_name] + plant_move_index
        else:
            move_index = starting_index[move_name]
        return move_index

    def get_name(self):
        return self.__class__.__name__

    def get_move_name(self):
        return self.__class__.__name__


class Grow(Move):
    def __init__(self, board: "Board", tree: Tree, to_tree: Tree, cost: int):
        super().__init__(board=board, tree=tree, tile=tree.tile, cost=cost)
        self.to_tree = to_tree

    def execute(self):
        self.board.grow_tree(self.tree, self.to_tree, self.cost)
        return self.board

    def get_move_name(self):
        return f"{self.__class__.__name__}_{self.tree.tile.notation}"


class Plant(Move):
    def __init__(self, board: "Board", from_tile: Tile = None, tile: Tile = None, tree: Tree = None, cost: int = None):
        super().__init__(board=board, tile=tile, from_tile=from_tile, tree=tree, cost=cost)
        self.cost = self.cost if self.cost is not None else PLANT_LP_COST
        self.from_tile = from_tile

    def execute(self):
        self.board.plant_tree(self.tile, self.tree, self.from_tile, self.cost)
        return self.board

    def get_move_name(self):
        return f"{self.__class__.__name__}_{self.tile.notation}"


class Buy(Move):
    def execute(self):
        self.board.buy_tree(self.tree, self.cost)
        return

    def get_move_name(self):
        return f"{self.__class__.__name__}_{self.tree.size}"


class Collect(Move):
    def __init__(self, board: "Board", tile: Tile = None, tree: Tree = None, cost: int = None):
        super().__init__(board=board, tile=tree.tile, tree=tree, cost=cost)
        self.cost = COLLECT_LP_COST

    def execute(self):
        self.board.collect_tree(self.tree, self.cost)
        return self.board

    def get_move_name(self):
        return f"{self.__class__.__name__}_{self.tree.tile.notation}"


class EndGo(Move):
    def __init__(self, board: "Board", player_number: int):
        super().__init__(board=board)
        self.player_number = player_number

    def execute(self):
        self.board.end_go(self.player_number)
        return self.board
