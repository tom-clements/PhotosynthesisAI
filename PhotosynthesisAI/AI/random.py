import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from .. import Game


class RandomAI(BaseAI):
    def pick_move(self, game: Game, available_moves: List[Move]) -> Move:
        return random.choice(available_moves)
