from abc import abstractmethod
from typing import List

from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.player import Player


class BaseAI(Player):
    def play_turn(self, board):
        moves = self.starting_moves(board) if board.round_number in [0, 1] else self.moves_available(board)
        move = self.pick_move(board, moves)
        self.move(move)

    @abstractmethod
    def pick_move(self, board: Board, availablle_moves: List[Move]) -> Move:
        pass
