from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

from PhotosynthesisAI.game.components import Tree, Tile
from PhotosynthesisAI.game.utils.constants import PLANT_LP_COST, COLLECT_LP_COST, TREES
from PhotosynthesisAI.game.utils.utils import time_function


class Move:
    def __init__(self, board: "Board", tile: Tile = None, tree: Tree = None, cost: int = None):
        self.board = board
        self.tile = tile
        self.tree = tree
        self.cost = cost
        self._set_move_id()

    # this function gets hit a lot - contributes to 6% of compute
    @time_function
    def _set_move_id(self):
        num_tile_actions = len(self.board.data.tiles)
        move_name = self.__class__.__name__
        tile_index = self.tile.index if self.tile else None
        tree_size = self.tree.size if self.tree else None
        self.move_id = self._get_move_index(move_name, num_tile_actions, tile_index, tree_size)

    @staticmethod
    @lru_cache(maxsize=None)
    @time_function
    def _get_move_index(move_name: str, num_tile_actions: int, tile_index: int, tree_size: int) -> int:
        num_buy_moves = len(TREES.keys())
        starting_index = {
            "Plant": 0,
            "Grow": num_tile_actions * 1,
            "Collect": num_tile_actions * 2,
            "Buy": num_tile_actions * 3,
            "EndGo": num_tile_actions * 3 + num_buy_moves,
        }
        if move_name in ["Plant", "Grow", "Collect"]:
            move_index = starting_index[move_name] + tile_index
        elif move_name == "Buy":
            move_index = starting_index[move_name] + tree_size
        else:
            move_index = starting_index[move_name]
        return move_index

    def get_name(self):
        return self.__class__.__name__

    def get_notation(self):
        if not self.tile:
            return
        self.tile.index

class Grow(Move):
    def __init__(self, board: "Board", tree: Tree, to_tree: Tree, cost: int):
        super().__init__(board, tree=tree, tile=tree.tile, cost=cost)
        self.to_tree = to_tree

    def execute(self):
        self.board.grow_tree(self.tree, self.to_tree, self.cost)
        return self.board


class Plant(Move):
    def __init__(self, board: "Board", tile: Tile = None, tree: Tree = None, cost: int = None):
        super().__init__(board, tile, tree, cost)
        self.cost = self.cost if self.cost is not None else PLANT_LP_COST

    def execute(self):
        self.board.plant_tree(self.tile, self.tree, self.cost)
        return self.board


class Buy(Move):
    def execute(self):
        self.board.buy_tree(self.tree, self.cost)
        return


class Collect(Move):
    def __init__(self, board: "Board", tile: Tile = None, tree: Tree = None, cost: int = None):
        super().__init__(board, tree.tile, tree, cost)
        self.cost = COLLECT_LP_COST

    def execute(self):
        self.board.collect_tree(self.tree, self.cost)
        return self.board


class EndGo(Move):
    def __init__(self, board: "Board", player_number: int):
        super().__init__(board)
        self.player_number = player_number

    def execute(self):
        self.board.end_go(self.player_number)
        return self.board
