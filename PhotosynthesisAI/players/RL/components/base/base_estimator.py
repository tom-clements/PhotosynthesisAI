import os
import pickle
from collections import defaultdict
from typing import List


import numpy as np
import PhotosynthesisAI
from PhotosynthesisAI.game.utils.utils import time_function

from logging import getLogger

logger = getLogger(__name__)


class Estimator:
    def __init__(self, total_num_actions: int, start_features: List, replay_length: int = 100):
        self.total_num_actions = total_num_actions
        self._initialize_model(total_num_actions, start_features)
        self.replay = dict(features=[], y=[])
        self.replay_length = replay_length
        self.current_replay_size = 0
        self.replay_n_iter = 1

    def _initialize_model(self, start_features):
        self.model = None

    @staticmethod
    def _features_to_model_input(state_features):
        return state_features

    @time_function
    def predict(self, state_features: List):
        features = self._features_to_model_input(state_features)
        predictions = self.model.predict(features)
        return np.squeeze(predictions)

    # @time_function
    # def update(self, a, features, y):
    #     features = self._features_to_model_input(features)
    #     self.models[a].fit(features, [y])

    @staticmethod
    def _get_default_path(name):
        return os.path.join(os.path.dirname(PhotosynthesisAI.__file__), "players", "RL", "saved_models", f"{name}.pkl")

    @time_function
    def pickle_model(self, name, path=None):
        path = path if path else self._get_default_path(name)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @time_function
    def load_model(self, name: str, path=None):
        path = path if path else self._get_default_path(name)
        with open(path, "rb") as f:
            self.model = pickle.load(f)
