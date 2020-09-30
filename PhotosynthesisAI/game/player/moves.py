from copy import deepcopy
from dataclasses import dataclass
from PhotosynthesisAI.game.components import Tree, Tile
from PhotosynthesisAI.game.utils.constants import PLANT_LP_COST, COLLECT_LP_COST


@dataclass
class Move:
    board: "Board"


@dataclass
class Grow(Move):
    from_tree: Tree
    to_tree: Tree
    cost: int

    def execute(self):
        self.board.grow_tree(self.from_tree, self.to_tree, self.cost)
        return self.board


@dataclass
class Plant(Move):
    tile: Tile
    tree: Tree
    cost: int = PLANT_LP_COST

    def execute(self):
        self.board.plant_tree(self.tile, self.tree, self.cost)
        return self.board


@dataclass
class Buy(Move):
    tree: Tree
    cost: int

    def execute(self):
        self.board.buy_tree(self.tree, self.cost)
        return


@dataclass
class Collect(Move):
    tree: Tree
    cost: int = COLLECT_LP_COST

    def execute(self):
        self.board.collect_tree(self.tree, self.cost)
        return self.board


@dataclass
class EndGo(Move):
    player_number: int

    def execute(self):
        self.board.end_go(self.player_number)
        return self.board
