import hexy as hx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from .constants import *
from .tree import Tree
from .tile import Tile


class Board:
    tile_coords = hx.get_spiral(np.array((0, 0, 0)), 1, BOARD_RADIUS)
    x_tiles = [i for i in tile_coords if i[0] == 0]
    y_tiles = [i for i in tile_coords if i[1] == 0]
    z_tiles = [i for i in tile_coords if i[2] == 0]
    suns = np.array([
        np.array([0, 1, -1]),  # N
        np.array([1, 0, -1]),  # NE
        np.array([1, -1, 0]),  # SE
        np.array([0, -1, 1]),  # S
        np.array([-1, 0, 1]),  # SW
        np.array([-1, 1, 0]),  # NW
    ])

    def __init__(self, num_players):
        self.trees = []
        for i, p in enumerate(range(num_players)):
            for tree_type in TREES.keys():
                tree = TREES[tree_type]
                self.trees += [Tree(owner=i + 1, size=tree['size'], bought=True, tree_type=tree_type) for j in
                               range(tree['starting'])]
                self.trees += [Tree(owner=i + 1, size=tree['size'], bought=False, cost=cost, tree_type=tree_type) for
                               j, cost in enumerate(tree['cost'])]
        self.trees = np.array(self.trees)
        self.round_number = 1
        self.pg_active = False
        self.sun_position = 5
        self.data = [Tile(tree=None, coords=coord, index=self.get_tile_index(coord)) for i, coord in
                     enumerate(self.tile_coords)]

    def start_round(self, players):
        self.rotate_sun()
        for tile in self.data:
            tile.is_locked = False
        for player in players:
            player.l_points += 1

    def is_game_over(self):
        return self.round_number == self.max_sun_rotaions

    def rotate_sun(self):
        self.sun_position = (self.sun_position + 1) % 6
        self.round_number += 1
        return

    @classmethod
    def get_surrounding_tiles(cls, tile, radius):
        return

    @classmethod
    def get_tile_index(cls, coord):
        # https://stackoverflow.com/questions/28312374/numpy-where-compare-arrays-as-a-whole
        condition = (cls.tile_coords[:, 0] == coord[0]) & (cls.tile_coords[:, 1] == coord[1]) & (
                    cls.tile_coords[:, 2] == coord[2])
        return int(np.where(condition)[0])

    #         @staticmethod
    #     def get_tile_index(tile):
    #         # https://stackoverflow.com/questions/28312374/numpy-where-compare-arrays-as-a-whole
    #         condition = (tiles[:,0] == tile[0]) & (tiles[:,1] == tile[1]) & (tiles[:,2] == tile[2])
    #         return int(np.where(condition)[0])

    def get_tile(tile):
        tile_index = self.get_tile_index(tile)
        return self.data[tile_index]

    def grow_tree(self, from_tree, to_tree):
        tile_index = self.get_tile_index(tile)
        self.data[tile_index] = new_tree
        self.data

    def plant_tree(self, tile, tree):
        if self.data[tile.index].tree:
            raise ValueError("There is already a tree here")
        self.data[tile.index].tree = tree
        tree.tile = tile

    def collect_tree(self, tile):
        tile_index = self.get_tile_index(tile)
        if self.data[tile_index].tree.size != 4:
            raise ValueError('Cannot harvest a tree that is not fully grown')
        self.data[tile_index] = Tile(Tree(owner=0, size=0), coords=tile, index=tile_index)

    def show(self):
        player_colors = {0: 'green', 1: 'red', 2: 'blue'}
        colors = [player_colors[tile.tree.owner] if tile.tree else "green" for tile in self.data]
        labels = [tile.tree.size if tile.tree else "" for tile in self.data]
        hcoord = [c[0] for c in self.tile_coords]
        vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in self.tile_coords]
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8.5, 8.5)
        #         ax.set_aspect('equal')
        # Add some coloured hexagons
        for x, y, c, l in zip(hcoord, vcoord, colors, labels):
            hexagon = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                     orientation=np.radians(30),
                                     facecolor=c, alpha=0.2, edgecolor='k')
            ax.add_patch(hexagon)
            # Add a text label
            ax.text(x, y + 0.2, l, ha='center', va='center', size=20)
        sun_coords = self.suns[self.sun_position] * 4
        sun_y_coord = 2 * np.sin(np.radians(60)) * (sun_coords[1] - sun_coords[2]) / 3
        ax.text(sun_coords[0], sun_y_coord, 'SUN', ha='center', va='center', size=20, color='orange')
        # Add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, c=colors, alpha=0.5)
        plt.axis('off')
        plt.show()