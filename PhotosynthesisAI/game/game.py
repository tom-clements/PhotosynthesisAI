import time
import logging
import random
from typing import List, Dict

from .components import Board
from PhotosynthesisAI.game.utils.constants import MAX_SUN_ROTATIONS
from .player import Player

logger = logging.getLogger("Game")
logging.basicConfig(level=logging.INFO)


class Game:
    def __init__(self, players: List[Player], verbose: bool = True):
        self.players = players
        self.board = Board(players)
        self.verbose = verbose
        for player in self.players:
            player.score = 0
            player.l_points = 0

    def play(self, show_board_each_turn: bool = False):
        start_time = time.time()
        while not self.is_game_over():
            self.set_player_order()
            for player in self.player_order:
                player.play_turn(self.board)
            self.board.end_round()
            if show_board_each_turn:
                self.board.show()
        end_time = time.time()
        if self.verbose:
            logger.info(f"Ran game in {int((end_time-start_time)*1000)}ms")
            for player in self.players:
                logger.info(f"Player {player.number} score: {player.score}")

    def show(self):
        self.board.show()

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
