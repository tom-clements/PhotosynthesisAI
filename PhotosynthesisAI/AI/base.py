from abc import abstractmethod
from typing import List

from PhotosynthesisAI import Game
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.player import Player


class BaseAI(Player):

    def __init__(self):
        super().__init__()

    def play_turn(self, game):
        moves = self.starting_moves(game.board) if game.board.round_number in [0, 1] else self.moves_available(game.board)
        move = self.pick_move(game, moves)
        game.execute_move(move)

    @abstractmethod
    def pick_move(self, game: Game, available_moves: List[Move]) -> Move:
        pass
