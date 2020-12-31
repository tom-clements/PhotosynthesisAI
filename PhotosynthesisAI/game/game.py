import time
import gc
import logging
import random
from typing import List, Dict
from copy import deepcopy

import pandas as pd

from .components import Board
from PhotosynthesisAI.game.utils.constants import MAX_SUN_ROTATIONS, TREES
from .player import Player
from .utils.utils import hash_text, time_function

logger = logging.getLogger("Game")
logging.basicConfig(level=logging.INFO)


class Game:
    def __init__(self, players: List[Player]):
        self.players = players
        self.round_turn = 0
        self.board = None
        self.total_num_actions = None
        self.set_initial_player_order()
        self.board = Board(self.players)
        self.total_num_actions = self._total_num_actions()
        for player in self.players:
            player.reset(self)
        self.board.reset()

    def player_turns(self):
        for i, player in enumerate(self.players):
            while player.go_active:
                player.play_turn(self)
                if self.board.round_number in [0, 1]:
                    break
            self.round_turn = i

    def play(self, verbose: bool = True, show_board_each_turn: bool = False):
        start_time = time.time()
        while not self.is_game_over():
            self.set_player_order()
            self.player_turns()
            self.board.end_round()
            if show_board_each_turn:
                self.board.show()
        end_time = time.time()
        if verbose:
            logger.info(f"Ran game in {int((end_time-start_time)*1000)}ms")
            for player in self.players:
                logger.info(f"Player {player.number} score: {player.score}")

    def show(self):
        self.board.show()

    def set_initial_player_order(self):
        # random.shuffle(self.players)
        for i, player in enumerate(self.players):
            player.number = i + 1
            player.go_order = i + 1

    @time_function
    def set_player_order(self):
        if self.board.round_number == 0:
            return
        # don't rotate until round 2 after 2 initial and first normal round
        elif self.board.round_number in [1, 2]:
            return
        else:
            self.players = self.players[1:] + [self.players[0]]

    def is_game_over(self):
        return self.board.round_number == (MAX_SUN_ROTATIONS * 6) + 2

    def get_score(self) -> Dict[Player, int]:
        return {player: player.score for player in self.board.data.players}

    def get_winner(self) -> List[Player]:
        if self.is_game_over():
            scores = self.get_score()
            max_score = max(scores.values())
            winners = [player for player, score in scores.items() if score == max_score]
            if len(winners) > 1:
                return []
            else:
                return winners

        else:
            raise ValueError('The game has not finished!')

    def _total_num_actions(self) -> int:
        num_tile_actions = len(self.board.data.tiles)
        num_buy_moves = len(TREES.keys())
        end_go_moves = 1
        # plant + grow + collect = 3 * num_tile_actions
        return num_tile_actions * 3 + num_buy_moves + end_go_moves

    @time_function
    def get_state(self) -> Dict:
        # remove metadata - maybe keep cost?
        tree_keys_to_remove = ['shadow', 'tile', 'cost', 'score', 'tree_type']
        tiles = [deepcopy(tile.__dict__) for tile in self.board.data.tiles]
        for tile in tiles:
            if tile['tree']:
                tile['owner'] = tile['tree'].owner
                tile['size'] = tile['tree'].size
            else:
                tile['owner'] = None
                tile['size'] = None
            tile.pop('tree')
        trees = [
            deepcopy(tree.__dict__)
            for tree in self.board.data.trees
        ]
        for t in trees:
            for k in tree_keys_to_remove:
                t.pop(k)
        trees = sorted(trees, key=lambda i: (i['is_bought'], i['size']))
        l_points = {player.number: player.l_points for player in self.players}
        scores = {player.number: player.score for player in self.players}
        round_number = self.board.round_number
        state = {
            'tiles': tiles,
            'trees': trees,
            'l_points': l_points,
            'scores': scores,
            'round_number': round_number,
            'turns_in_round': self.round_turn
        }
        return state

    @time_function
    def get_partial_state(self) -> Dict:
        tile_keys_to_remove = ['coords', 'richness', 'is_locked', 'is_shadow']
        tiles = [deepcopy(tile.__dict__) for tile in self.board.data.tiles]
        for tile in tiles:
            if tile['tree']:
                tile['owner'] = tile['tree'].owner
                # tile['size'] = tile['tree'].size
            else:
                tile['owner'] = None
                tile['size'] = None
            tile.pop('tree')
            for k in tile_keys_to_remove:
                tile.pop(k)
        round_number = self.board.round_number
        state = {
            'tiles': tiles,
            # 'round_number': round_number,
        }
        return state

    @time_function
    def get_linear_features(self, player) -> List:
        opponent = [p for p in self.players if p.number != player.number][0]
        next_go_board = deepcopy(self.board)
        next_go_board.rotate_sun()
        next_go_board._set_shadows()
        # does not work with more than 2 players
        toggle = -1 if player.number == 1 else 1
        # tiles = {
        #     f"tile{tile.index}": (toggle*(player.number*2 - 3) * (tile.tree.size + 1) if tile.tree else 0)
        #     for player in self.players for tile in self.board.data.tiles
        # }
        tiles = [int(t.is_shadow) for t in next_go_board.data.tiles]
        num_seeds_owned = len([tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 0])
        num_small_trees_owned = len([tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 1])
        num_medium_trees_owned = len([tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 2])
        num_large_trees_owned = len([tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 3])
        opp_num_seeds_owned = len([tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 0])
        opp_num_small_trees_owned = len([tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 1])
        opp_num_medium_trees_owned = len([tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 2])
        opp_num_large_trees_owned = len([tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 3])
        l_points = player.l_points
        opponent_l_points = - opponent.l_points
        score = player.score
        opponent_score = - opponent.score
        round_number = self.board.round_number
        round_turn_number = self.round_turn
        features = tiles + [
            # num_seeds_owned,
            # num_small_trees_owned,
            # num_medium_trees_owned,
            # num_large_trees_owned,
            # opp_num_seeds_owned,
            # opp_num_small_trees_owned,
            # opp_num_medium_trees_owned,
            # opp_num_large_trees_owned,
            # l_points,
            # opponent_l_points,
            # score,
            # opponent_score,
            round_number,
            round_turn_number,
        ]
        return features

    @time_function
    def execute_move(self, move):
        move.execute()

    @time_function
    def get_nn_features(self, player) -> List:
        opponent = [p for p in self.players if p.number != player.number][0]
        # next_go_board = deepcopy(self.board)
        # next_go_board.rotate_sun()
        # next_go_board._set_shadows()
        # does not work with more than 2 players
        # tiles = {
        #     f"tile{tile.index}": (toggle*(player.number*2 - 3) * (tile.tree.size + 1) if tile.tree else 0)
        #     for player in self.players for tile in self.board.data.tiles
        # }
        # shadows = [int(t.is_shadow) for t in next_go_board.data.tiles]
        sizes = [(int(t.tree.size) + 1 if t.tree else 0) for t in self.board.data.tiles]
        owners = [(int(t.tree.owner) if t.tree else 0) for t in self.board.data.tiles]
        num_seeds_owned = len(
            [tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 0])
        num_small_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 1])
        num_medium_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 2])
        num_large_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[player.number]['bought'].values() if tree.size == 3])
        opp_num_seeds_owned = len(
            [tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 0])
        opp_num_small_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 1])
        opp_num_medium_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 2])
        opp_num_large_trees_owned = len(
            [tree for tree in self.board.tree_of_trees[opponent.number]['bought'].values() if tree.size == 3])
        l_points = player.l_points
        opponent_l_points = - opponent.l_points
        score = player.score
        opponent_score = - opponent.score
        round_number = self.board.round_number
        round_turn_number = self.round_turn
        features = sizes + owners + [
            num_seeds_owned,
            num_small_trees_owned,
            num_medium_trees_owned,
            num_large_trees_owned,
            opp_num_seeds_owned,
            opp_num_small_trees_owned,
            opp_num_medium_trees_owned,
            opp_num_large_trees_owned,
            l_points,
            opponent_l_points,
            score,
            opponent_score,
            round_number,
            round_turn_number,
        ]
        return features

    def execute_move(self, move):
        move.execute()

