import hexy as hx
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.patches import RegularPolygon
from PhotosynthesisAI.game.utils.constants import BOARD_RADIUS, MAX_SUN_ROTATIONS, TOKENS, TREES
from PhotosynthesisAI.game.components import Tile, Tree, Token
from PhotosynthesisAI.game.utils.utils import find_array_in_2D_array
from PhotosynthesisAI.game.player import Player
from typing import List, Union


class Board:
    tile_coords = hx.get_spiral(np.array((0, 0, 0)), 1, BOARD_RADIUS)
    x_tiles = [i for i in tile_coords if i[0] == 0]
    y_tiles = [i for i in tile_coords if i[1] == 0]
    z_tiles = [i for i in tile_coords if i[2] == 0]
    suns = np.array(
        [
            np.array([0, 1, -1]),  # N
            np.array([1, 0, -1]),  # NE
            np.array([1, -1, 0]),  # SE
            np.array([0, -1, 1]),  # S
            np.array([-1, 0, 1]),  # SW
            np.array([-1, 1, 0]),  # NW
        ]
    )

    def __init__(self, players: List[Player]):
        trees = []
        for i, p in enumerate(players):
            for tree_type in TREES.keys():
                tree = TREES[tree_type]
                trees += [
                    Tree(owner=i + 1, size=tree["size"], is_bought=True, tree_type=tree_type, score=tree["score"])
                    for j in range(tree["starting"])
                ]
                trees += [
                    Tree(
                        owner=i + 1,
                        size=tree["size"],
                        is_bought=False,
                        cost=cost,
                        tree_type=tree_type,
                        score=tree["score"],
                    )
                    for j, cost in enumerate(tree["cost"])
                ]
        self.round_number = 0
        self.pg_active = False
        self.sun_position = 5
        Data = namedtuple("data", "tiles trees players tokens")
        self.data = Data(
            tiles=[
                Tile(tree=None, coords=coord, index=self.get_tile_index(coord))
                for i, coord in enumerate(self.tile_coords)
            ],
            trees=np.array(trees),
            players=players,
            tokens=[Token(richness=key, value=value) for key in TOKENS for value in TOKENS[key]],
        )

    #########
    # UTILS #
    #########

    def _get_tile_from_coords(self, coords) -> Tile:
        # can maybe cache this calcualion, although it is quite quick
        mask = find_array_in_2D_array(coords, self.tile_coords)
        return np.array(self.data.tiles)[mask][0]

    def is_game_over(self):
        return self.round_number == (MAX_SUN_ROTATIONS * 6) + 1

    def get_surrounding_tiles(self, tile: Tile, radius: int) -> List[Tile]:
        surrounding_tile_coords = tile.get_surrounding_coords(radius)
        # subset to tiles in board and exclude the tile itself
        tiles = [
            t
            for t in self.data.tiles
            if any(find_array_in_2D_array(t.coords, surrounding_tile_coords)) & (not all(t.coords == tile.coords))
        ]
        return tiles

    @classmethod
    def get_tile_index(cls, coord):
        mask = find_array_in_2D_array(coord, cls.tile_coords)
        return int(np.where(mask)[0])

    def get_empty_unlocked_tiles(self) -> List[Tile]:
        return [tile for tile in self.data.tiles if (not tile.tree) & (not tile.is_locked)]

    #########

    ######################
    # ROUND CALCULATIONS #
    ######################

    def end_round(self):
        for player in self.data.players:
            player.go_active = True
        self.rotate_sun()
        self._set_shadows()
        for tile in self.data.tiles:
            tile.is_locked = False
            if (not tile.is_shadow) & (tile.tree is not None):
                player = [player for player in self.data.players if player.number == tile.tree.owner][0]
                player.l_points += tile.tree.score

    def get_player_order(self) -> List[Player]:
        num_players = len(self.data.players)
        first_player_number = self.round_number % num_players
        order = [1 + ((self.round_number + i) % num_players) for i in range(num_players)]

        player_order = []
        for i in order:
            player_order.append([player for player in self.data.players if player.number == i][0])
        return player_order

    def rotate_sun(self):
        self.sun_position = (self.sun_position + 1) % 6
        self.round_number += 1
        return

    def _get_tiles_at_sun_edge(self) -> List[Tile]:
        sun = self.suns[self.sun_position]
        mask = (
            np.count_nonzero(
                (np.array([np.array(sun) for t in self.tile_coords]) == self.tile_coords / 3) * np.array(sun), axis=1
            )
            > 0
        )
        edge_coords = self.tile_coords[mask]
        tiles = [self._get_tile_from_coords(coords) for coords in edge_coords]
        return tiles

    def _get_tiles_along_same_axis(self, start_tile: Tile, axis: Union[List, np.ndarray]) -> List[Tile]:
        coords = start_tile.coords
        tiles = []
        while max(abs(coords)) <= BOARD_RADIUS:
            tiles.append(self._get_tile_from_coords(coords))
            coords = coords - axis
        return tiles

    def _set_shadows_along_axis(self, tile: Tile, axis: np.ndarray):
        axis_tiles = self._get_tiles_along_same_axis(tile, axis)
        current_shadow_size = 0
        for axis_tile in axis_tiles:
            axis_tile.is_shadow = True if current_shadow_size > 0 else False
            current_shadow_size = (current_shadow_size - 1) if current_shadow_size > 0 else 0
            if not axis_tile.tree:
                continue
            current_shadow_size = max([axis_tile.tree.shadow, current_shadow_size])

    def _set_shadows(self):
        sun = self.suns[self.sun_position]
        edge_tiles = self._get_tiles_at_sun_edge()
        for tile in edge_tiles:
            self._set_shadows_along_axis(tile, sun)

    ######################

    ##################
    # MOVE EXECUTION #
    ##################

    def get_next_token(self, richness: int) -> Token:
        # get the first token
        while richness > 0:
            tokens = [token for token in self.data.tokens if (token.richness == richness) & (token.owner is None)]
            if tokens:
                return tokens[0]
            richness -= 1
        raise ValueError('Ran out of tokens')

    def grow_tree(self, from_tree: Tree, to_tree: Tree, cost: int):
        # should wrap these into unit tests
        if not from_tree.tile:
            raise ValueError("This tree isn't on the board")
        if from_tree.size == 3:
            raise ValueError("This tree can't grow anymore!")
        if (to_tree.size - from_tree.size) != 1:
            raise ValueError("The tree is trying to grow to a size that isn't one bigger!")
        if not to_tree.is_bought:
            raise ValueError("The tree hasn't been bought yet")
        if to_tree.tile:
            raise ValueError("The tree growing to is already on the board")
        tile = from_tree.tile
        to_tree.tile = tile
        from_tree.tile = None
        tile.tree = to_tree
        tile.is_locked = True
        player = [player for player in self.data.players if player.number == tile.tree.owner][0]
        player.l_points -= cost

    def plant_tree(self, tile: Tile, tree: Tree, cost: int):
        if tile.tree:
            raise ValueError("There is already a tree here")
        if tree.tile:
            raise ValueError("This tree is already planted")
        if not tree.is_bought:
            raise ValueError("The tree hasn't been bought yet")
        tile.tree = tree
        tree.tile = tile
        self.data.tiles[tile.index].tree = tree
        tree.tile = tile
        tile.is_locked = True
        player = [player for player in self.data.players if player.number == tile.tree.owner][0]
        player.l_points -= cost

    def collect_tree(self, tree: Tree, cost: int):
        if not tree.tile:
            raise ValueError("This tree isn't on the board")
        if tree.size != 3:
            raise ValueError("The tree isn't fully grown")
        tile = tree.tile
        tile.tree = None
        tree.tile = None
        tree.is_bought = False
        tile.is_locked = True
        player = [player for player in self.data.players if player.number == tree.owner][0]
        player.l_points -= cost
        token = self.get_next_token(tile.richness)
        token.owner = tree.owner
        player.score += token.value

    def buy_tree(self, tree: Tree, cost: int):
        if tree.is_bought:
            raise ValueError("The tree has already been bought")
        tree.is_bought = True
        player = [player for player in self.data.players if player.number == tree.owner][0]
        player.l_points -= cost

    def end_go(self, player_number):
        player = [player for player in self.data.players if player.number == player_number][0]
        player.go_active = False
        return True

    ##################

    ##################
    # VISUALISAIION #
    ##################

    def show(self):
        player_colors = {0: "green", 1: "red", 2: "blue"}
        colors = [player_colors[tile.tree.owner] if tile.tree else "green" for tile in self.data.tiles]
        shadows = ["black" if tile.is_shadow else "orange" for tile in self.data.tiles]
        labels = [tile.tree.size if tile.tree else "" for tile in self.data.tiles]
        hcoord = [c[0] for c in self.tile_coords]
        vcoord = [2.0 * np.sin(np.radians(60)) * (c[1] - c[2]) / 3.0 for c in self.tile_coords]

        fig, ax = plt.subplots(1)
        fig.set_size_inches(12.5, 8.5)
        plt.axis([-5, 10, -5, 5])
        ax.set_aspect('equal')
        # Add some coloured hexagons
        for x, y, c, l in zip(hcoord, vcoord, colors, labels):
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=2.0 / 3.0,
                orientation=np.radians(30),
                facecolor=c,
                alpha=0.2,
                edgecolor="k",
            )
            ax.add_patch(hexagon)
            # Add a text label
            ax.text(x, y + 0.3, l, ha="center", va="center", size=10)
        sun_coords = self.suns[self.sun_position] * 4
        sun_y_coord = 2 * np.sin(np.radians(60)) * (sun_coords[1] - sun_coords[2]) / 3
        ax.text(sun_coords[0], sun_y_coord, "SUN", ha="center", va="center", size=20, color="orange")

        for player in self.data.players:
            ax.text(
                7,
                2 - player.number/2,
                f"Player {player.number}: Light Points: {player.l_points}, score: {player.score}",
                ha="center",
                va="center",
                size=10,
                color="black",
            )

        ax.text(7, 4, f"Round number {self.round_number}", ha="center", va="center", size=10, color="black")

        # Add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, c=shadows, alpha=0.5)
        plt.axis("off")
        plt.show()

    ###################
