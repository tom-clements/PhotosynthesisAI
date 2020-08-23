import time
import logging
from typing import List

from .components import Board
from .player import Player

logger = logging.getLogger('Game')
logging.basicConfig(level=logging.INFO)


class Game:

    def __init__(self, players: List[Player]):
        self.players = players
        self.board = Board(players)

    def play(self, show_board_each_turn: bool = False):
        start_time = time.time()
        while not self.board.is_game_over():
            player_order = self.board.get_player_order()
            for player in player_order:
                player.play_turn(self.board)
            self.board.end_round()
            if show_board_each_turn:
                self.board.show()
        end_time = time.time()
        logger.info(f'Ran game in {int((end_time-start_time)*1000)}ms')
        for player in self.players:
            logger.info(f'Player {player.number} score: {player.score}')

    def show(self):
        self.board.show()
