import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move


class RandomAI(BaseAI):
    def pick_move(self, board: Board, availablle_moves: List[Move]) -> Move:
        return random.choice(availablle_moves)
