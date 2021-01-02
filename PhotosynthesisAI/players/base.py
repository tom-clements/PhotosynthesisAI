from abc import abstractmethod
from typing import List

from PhotosynthesisAI import Game
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.player import Player


class BaseAI(Player):
    def __init__(self, name: str = None):
        super().__init__(name)

    @abstractmethod
    def pick_move(self, game: Game, available_moves: List[Move]) -> Move:
        pass
