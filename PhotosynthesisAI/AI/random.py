import random

from .base import BaseAI


class RandomAI(BaseAI):
    def pick_move(self, board, moves):
        return random.choice(moves)
