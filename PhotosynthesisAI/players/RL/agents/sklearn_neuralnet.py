import gc
from collections import defaultdict
from typing import List


import numpy as np
from sklearn.neural_network import MLPRegressor
from PhotosynthesisAI import Game
from PhotosynthesisAI.players.RL.components.ML.neural_network import NNRegressor, ActFns
from PhotosynthesisAI.players.RL.components.base.base_estimator import Estimator
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.utils.utils import time_function
from PhotosynthesisAI.players.RL.components.base.base_rl import BaseRL

from logging import getLogger


logger = getLogger(__name__)


class SKlearnNNEstimator(Estimator):
    def initialize_model(self, total_num_actions: int, start_features):
        model = MLPRegressor(hidden_layer_sizes=(1024,), learning_rate="constant")
        model.partial_fit(self._features_to_model_input(start_features), np.zeros((1, total_num_actions)))
        self.model = model

    @time_function
    def _features_to_model_input(self, state_features):
        return [state_features]

    def update(self, features, y):
        self.replay["features"].append(features)
        self.replay["y"].append(y)
        self.current_replay_size += 1
        if self.current_replay_size == self.replay_length:
            features = self.replay["features"]
            y = np.array(self.replay["y"])
            for i in range(self.replay_n_iter):
                self.model.partial_fit(features, y)
            self.current_replay_size = 0
            self.replay = dict(features=[], y=[])


class SKlearnNNAI(BaseRL):
    def __init__(self, epsilon: float = 0.1, load_model=False, name: str = "nn", train: bool = True):
        super().__init__(name)
        self.epsilon = epsilon
        self.policy = None
        self.estimator = None
        self.rewards = []
        self.discount_factor = 1
        self.load_model = load_model
        self.state = None
        self.name = name
        self.train = train
        self.a = 1

    def _set_estimator(self, game: Game):
        if not self.estimator:
            self.estimator = SKlearnNNEstimator(
                game.total_num_actions, game.get_nn_features(self), load_model=self.load_model, name=self.name
            )
