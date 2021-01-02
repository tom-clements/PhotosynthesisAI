import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from .. import Game


class RandomAI(BaseAI):
    def play_move(self, game: Game, moves: List[Move]) -> Move:
        move = random.choice(moves)
        game.execute_move(move)
