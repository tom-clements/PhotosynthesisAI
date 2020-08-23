from typing import List
from PhotosynthesisAI.game.utils.constants import TREES, PLANT_LP_COST, COLLECT_LP_COST
from .moves import Plant, Grow, Collect, Buy, EndGo
from ..components.tree import Tree
from .moves import Move


class Player:
    def __init__(self, number):
        self.number = number
        self.l_points = 0
        self.score = 0
        self.go_active = True

    def starting_moves(self, board: 'Board') -> List[Plant]:
        free_tiles = [tile for tile in board.data.tiles if (3 in abs(tile.coords)) & (not tile.tree)]
        starting_tree = [tree for tree in board.data.trees if (tree.owner == self.number) & (tree.size == 1)][0]
        return [Plant(board=board, tile=tile, tree=starting_tree, cost=0) for tile in free_tiles]

    def get_planting_moves(self, board: 'Board', trees_on_board: List[Tree], trees_bought: List[Tree]) -> List[Plant]:
        # check is can afford to plant
        if self.l_points < PLANT_LP_COST:
            return []

        seeds = [tree for tree in trees_bought if tree.size == TREES["seed"]["size"]]
        # check if any seeds banked
        if not seeds:
            return []

        # check radius around trees to plant
        surrounding_tiles = [board.get_surrounding_tiles(tree.tile, tree.size) for tree in trees_on_board]
        # flatten
        surrounding_tiles = [item for sublist in surrounding_tiles for item in sublist]
        # deduplicate
        surrounding_tiles = list(set(surrounding_tiles))
        # can only plant in empty unlocked tiles
        empty_unlocked_tiles = board.get_empty_unlocked_tiles()
        surrounding_tiles = [tile for tile in surrounding_tiles if tile in empty_unlocked_tiles]
        # only use seed[0] as no point duplicating moves for all free seeds
        moves = [Plant(board=board, tile=tile, tree=seeds[0]) for tile in surrounding_tiles]
        return moves

    def get_growing_moves(self, board: 'Board', trees_on_board: List[Tree], trees_bought: List[Tree]) -> List[Grow]:
        if self.l_points == 0:
            return []

        trees_to_grow = [
            tree
            for tree in trees_on_board
            if (tree.size in [TREES["seed"]["size"], TREES["small"]["size"], TREES["medium"]["size"]])
            & (not tree.tile.is_locked)
        ]
        growing_trees = [
            tree
            for tree in trees_bought
            if tree.size in [TREES["small"]["size"], TREES["medium"]["size"], TREES["large"]["size"]]
        ]
        if (len(trees_to_grow) == 0) | (len(growing_trees) == 0):
            return []

        # TODO: could not use nested for loop for this
        moves = []
        for tree in trees_to_grow:
            for g_tree in growing_trees:
                if (g_tree.size - tree.size) == 1:
                    # check if can afford
                    if self.l_points < tree.size:
                        continue
                    moves.append(Grow(board=board, from_tree=tree, to_tree=g_tree, cost=tree.size))
                    # there is no point having multiple moves for growing to all available trees, so break early
                    break
        return moves

    def get_collecting_moves(self, board: 'Board', trees_on_board: List[Tree]) -> List[Collect]:
        if self.l_points < COLLECT_LP_COST:
            return []
        trees_to_collect = [tree for tree in trees_on_board if (tree.size == 3) & (not tree.tile.is_locked)]
        moves = [Collect(board=board, tree=tree) for tree in trees_to_collect]
        return moves

    def get_buying_moves(self, board: 'Board', trees_bought: List[Tree], trees_in_shop: List[Tree]) -> List[Buy]:
        bought_tree_sizes = list(set([tree.size for tree in trees_bought]))
        moves = [
            Buy(board=board, tree=tree, cost=tree.cost)
            for tree in trees_in_shop
            if (tree.cost <= self.l_points) & (tree.size not in bought_tree_sizes)
        ]
        return moves

    def moves_available(self, board: 'Board') -> List[Move]:
        trees_bought = [
            tree for tree in board.data.trees if (tree.owner == self.number) & (not tree.tile) & tree.is_bought
        ]
        trees_in_shop = [
            tree for tree in board.data.trees if (tree.owner == self.number) & (not tree.tile) & (not tree.is_bought)
        ]
        trees_on_board = [tree for tree in board.data.trees if (tree.owner == self.number) & (tree.tile is not None)]
        planting_moves = self.get_planting_moves(board, trees_on_board, trees_bought)
        growing_moves = self.get_growing_moves(board, trees_on_board, trees_bought)
        collecting_moves = self.get_collecting_moves(board, trees_on_board)
        buying_moves = self.get_buying_moves(board, trees_bought, trees_in_shop)
        return (
            planting_moves
            + growing_moves
            + collecting_moves
            + buying_moves
            + [EndGo(board=board, player_number=self.number)]
        )

    def move(self, move):
        move.execute()

    def play_turn(self, board):
        return
