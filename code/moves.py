from dataclasses import dataclass
from .board import Board
from .tree import Tree
from .tile import Tile
from .constants import PLANT_LP_COST, COLLECT_LP_COST


@dataclass
class Move:
    board: Board

@dataclass
class Grow(Move):
    from_tree: Tree
    to_tree: Tree
    cost: int

    def execute(self):
        self.board.grow_tree(self.from_tree, self.to_tree)
        return self.board

@dataclass
class Plant(Move):
    tile: Tile
    tree: Tree
    cost: int = PLANT_LP_COST

    def execute(self):
        self.board.plant_tree(self.tile, self.tree)
        return self.board


@dataclass
class Buy(Move):
    tree: Tree
    cost: int = 0

    def execute(self):
        return


@dataclass
class Collect(Move):
    tree: Tree
    cost: int = COLLECT_LP_COST

    def execute(self):
        self.board.plant_tree(self.tree)
        return self.board


@dataclass
class EndGo(Move):
    cost: int = 0

    def execute(self):
        return self.board
