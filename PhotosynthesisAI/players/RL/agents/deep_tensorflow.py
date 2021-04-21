import gc
from collections import defaultdict
from typing import List


import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from PhotosynthesisAI import Game
from PhotosynthesisAI.players.RL.components.ML.neural_network import NNRegressor, ActFns
from PhotosynthesisAI.players.RL.components.base.base_estimator import Estimator
from PhotosynthesisAI.players.RL.components.environment.environment import WinRewardEnvironment
from PhotosynthesisAI.players.RL.components.policy.epislon_greedy import make_epsilon_greedy_policy
from PhotosynthesisAI.players.base import BaseAI
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.utils.utils import time_function

from logging import getLogger

logger = getLogger(__name__)


class TensorflowNNEstimator(Estimator):
    def _initialize_model(self, total_num_actions: int, start_features):
        model = models.Sequential()
        # Feature Extraction Section (The Convolution and The Pooling Layer)
        # model.add(layers.Conv1D(filters=6, kernel_size=(5,), activation='relu', input_dim=len(start_features)))
        # model.add(layers.AveragePooling1D())
        # model.add(layers.Conv1D(filters=16, kernel_size=(5,), activation='relu'))
        # model.add(layers.AveragePooling1D())
        # Reshape the image into one-dimensional vector
        # model.add(layers.Flatten())
        # Classification Section (The Fully Connected Layer)
        # model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation="relu", input_dim=len(start_features)))
        model.add(layers.Dense(total_num_actions))
        model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"])
        model.train_on_batch(x=self._features_to_model_input([start_features]), y=np.zeros((1, total_num_actions)))
        self.model = model

    @time_function
    def _features_to_model_input(self, state_features):
        return np.array(state_features)
        state_features = state_features[0]
        print(np.array(state_features, dtype=float).reshape((1, len(state_features))).shape)
        return np.array(state_features, dtype=float).reshape((1, len(state_features)))

    @time_function
    def predict(self, state_features: List):
        features = self._features_to_model_input([state_features])
        predictions = self.model.predict(features)
        return np.squeeze(predictions)

    def update(self, features, y):
        self.replay["features"].append(features)
        self.replay["y"].append(y)
        self.current_replay_size += 1
        if self.current_replay_size == self.replay_length:
            features = self._features_to_model_input(self.replay["features"])
            y = np.array(self.replay["y"])
            for i in range(self.replay_n_iter):
                self.model.train_on_batch(features, y)
            self.current_replay_size = 0
            self.replay = dict(features=[], y=[])
        return


class DeepAI(BaseAI):
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

    def initialise(self, game: Game):
        self._set_estimator(game)
        self._set_policy(game)
        self._set_environment(game)
        self.rewards = []
        if self.load_model:
            self.estimator.load_model(self.name)

    def _set_estimator(self, game: Game):
        if not self.estimator:
            self.estimator = TensorflowNNEstimator(game.total_num_actions, game.get_nn_features(self))

    def _set_environment(self, game: Game):
        self.env = WinRewardEnvironment(game=game, player=self)

    def _set_policy(self, game: Game):
        if not self.policy:
            self.policy = make_epsilon_greedy_policy(self.estimator, self.epsilon, game.total_num_actions)

    @staticmethod
    @time_function
    def _action_to_move(action: int, available_moves: List[Move]) -> Move:
        move = [move for move in available_moves if move.move_id == action][0]
        return move

    @time_function
    def _get_policy_action(self, game, available_actions):
        probabilities = self.policy(self.state, available_actions=available_actions)
        action = np.random.choice(game.total_num_actions, p=probabilities)

        return action

    @time_function
    def get_td_target(self, new_state, reward, available_actions=None):
        # The Q-learning target policy is a greedy one, hence the `max`
        q_values_new_state = self.estimator.predict(state_features=new_state)
        target = np.copy(q_values_new_state)

        # set value of invalid actions to be 0
        if available_actions:
            mask = np.ones(len(q_values_new_state), np.bool)
            mask[available_actions] = 0
            q_values_new_state[mask] = np.nan

        # update action q value
        update = reward + self.discount_factor * np.nanmax(q_values_new_state)
        target[np.nanargmax(q_values_new_state)] += update
        return target

    @time_function
    def play_move(self, game: Game, moves: List[Move]):
        available_actions = [move.move_id for move in moves]
        self.state = game.get_nn_features(self)
        if len(available_actions) == 1:
            self.action = available_actions[0]
        else:
            self.action = self._get_policy_action(game, available_actions)
        move = self._action_to_move(self.action, moves)
        # Perform the action -> Get the reward and observe the next state
        new_state, reward = self.env.step(move)
        if self.train:
            td_target = self.get_td_target(new_state, reward, available_actions)
            self.estimator.update(self.state, td_target)

    def game_ended(self, game):
        if self.train:
            self.state = game.get_nn_features(self)
            reward = self.env.get_reward_game()
            q_values_state = self.estimator.predict(state_features=self.state) * 0 + reward
            self.estimator.update(self.state, q_values_state)

    def save_progress(self):
        self.save_model()

    @time_function
    def save_model(self):
        self.estimator.pickle_model(self.name)

    @time_function
    def load_model(self):
        self.estimator.load_model(self.name)
