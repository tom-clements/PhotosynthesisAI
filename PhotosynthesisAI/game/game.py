import time
import logging
import random
from typing import List, Dict
from copy import deepcopy

import pandas as pd

from .components import Board
from PhotosynthesisAI.game.utils.constants import MAX_SUN_ROTATIONS
from .player import Player
from .utils.utils import hash_text, time_function

logger = logging.getLogger("Game")
logging.basicConfig(level=logging.INFO)


class Game:
    def __init__(self, players: List[Player]):
        self.players = players
        self.board = Board(players)
        self.round_turn = 0
        self.states = []
        for player in self.players:
            player.score = 0
            player.l_points = 0

    def player_turns(self):
        for i, player in enumerate(self.player_order):
            while player.go_active:
                player.play_turn(self.board)
                # self.states.append(hash_text(str(self.get_partial_state())))
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

    @time_function
    def set_player_order(self):
        if self.board.round_number == 0:
            player_order = self.players
            random.shuffle(player_order)
            self.player_order = player_order
        # don't rotate until round 2 after 2 initial and first normal round
        elif self.board.round_number in [1, 2]:
            return
        else:
            self.player_order = self.player_order[1:] + [self.player_order[0]]

    def is_game_over(self):
        return self.board.round_number == (MAX_SUN_ROTATIONS * 6) + 2

    def get_score(self) -> Dict[Player, int]:
        return {player: player.score for player in self.board.data.players}

    def get_winner(self) -> List[Player]:
        if self.is_game_over():
            scores = self.get_score()
            max_score = max(scores.values())
            return [player for player, score in scores.items() if score == max_score]

        else:
            raise ValueError('The game has not finished!')

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