from copy import deepcopy

from PhotosynthesisAI import Game
from PhotosynthesisAI.game import Player
from PhotosynthesisAI.game.utils.utils import time_function


class Environment:

    def __init__(self, game: Game, player: Player):
        self.game = game
        self.player = player

    def _get_reward(self, start_player):
        pass

    @time_function
    def step(self, move):
        start_player_stats = {'points': self.player.l_points, 'score': self.player.score}
        self.game.execute_move(move)
        reward = self._get_reward(start_player_stats)
        new_state = self.game.get_nn_features(self.player)
        return new_state, reward