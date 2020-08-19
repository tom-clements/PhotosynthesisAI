import hexy as hx
import numpy as np
from .constants import TREES, PLANT_LP_COST, COLLECT_LP_COST
from .moves import Plant, Grow, Collect, Buy


class Player:
    def __init__(self, number):
        self.number = number
        self.l_points = 0
        self.score = 0

    def starting_moves(self, board):
        free_tiles = [tile for tile in board.data if (3 in abs(tile.coords)) & (not tile.tree)]
        starting_tree = [tree for tree in board.trees if (tree.owner == self.number) & (tree.size == 1)][0]
        return [Plant(board, tile, starting_tree) for tile in free_tiles]

    def get_planting_moves(self, board, trees_on_board, trees_bought, free_tiles):
        # check is can afford to plant
        if self.l_points < PLANT_LP_COST:
            return []

        seeds = [tree for tree in trees_bought if tree.size == TREES['seed']['size']]
        # check if any seeds banked
        if not seeds:
            return []

        tiles_to_plant = np.empty(shape=(0, 3))
        for tree in trees_on_board:
            # check radius around trees to plant
            for coords in hx.get_disk(tree.tile.coords, tree.size):

                # the above returns all the coordinates around a hex, filter to ones in board
                # https://stackoverflow.com/questions/28312374/numpy-where-compare-arrays-as-a-whole
                in_board = len(np.where(
                    (board.tile_coords[:, 0] == coords[0]) & (board.tile_coords[:, 1] == coords[1]) & (
                                board.tile_coords[:, 2] == coords[2]))[0]) > 0

                # de deuplicate
                if len(tiles_to_plant) > 0:
                    duplicate = len(np.where(
                        (tiles_to_plant[:, 0] == coords[0]) & (tiles_to_plant[:, 1] == coords[1]) & (
                                    tiles_to_plant[:, 2] == coords[2]))[0]) > 0
                else:
                    duplicate = False
                if in_board & (not duplicate):
                    tiles_to_plant = np.append(tiles_to_plant, np.array([coords]), axis=0)

        # only use seed[0] as no point duplicating moves for all free seeds
        moves = [Plant(board=board, tile=tile, tree=seeds[0]) for tile in tiles_to_plant]
        return moves

    def get_growing_moves(self, board, trees_on_board, trees_bought):
        if self.l_points == 0:
            return []

        trees_to_grow = [tree for tree in trees_on_board if
                         (tree.size in [TREES['seed']['size'], TREES['small']['size'], TREES['medium']['size']]) & (
                             not tree.tile.is_locked)]
        growing_trees = [tree for tree in trees_bought if
                         tree.size in [TREES['small']['size'], TREES['medium']['size'], TREES['large']['size']]]
        if (len(trees_to_grow) == 0) | (len(growing_trees) == 0):
            return []
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

    def get_collecting_moves(self, board, trees_on_board):
        if self.l_points < COLLECT_LP_COST:
            return []
        trees_to_collect = [tree for tree in trees_on_board if (tree.size == 3) & (not tree.tile.is_locked)]
        moves = [Collect(board=board, tree=tree) for tree in trees_to_collect]
        return moves

    def get_buying_moves(self, board, trees_in_shop):
        moves = [Buy(board=board, tree=tree, cost=tree.cost) for tree in trees_in_shop if tree.cost <= self.l_points]
        return moves

    def moves_available(self, board):
        trees_bought = [tree for tree in board.trees if (tree.owner == self.number) & (not tree.tile) & (tree.bought)]
        trees_in_shop = [tree for tree in board.trees if
                         (tree.owner == self.number) & (not tree.tile) & (not tree.bought)]
        trees_on_board = [tree for tree in board.trees if (tree.owner == self.number) & (not tree.tile is None)]
        free_tiles = [tile for tile in board.data if (not tile.tree) & (not tile.is_locked)]

        planting_moves = self.get_planting_moves(board, trees_on_board, trees_bought, free_tiles)
        growing_moves = self.get_growing_moves(board, trees_on_board, trees_bought)
        collecting_moves = self.get_collecting_moves(board, trees_on_board)
        buying_moves = self.get_buying_moves(board, trees_in_shop)
        return planting_moves + growing_moves + collecting_moves + buying_moves

    def move(self, move):
        move.execute()
