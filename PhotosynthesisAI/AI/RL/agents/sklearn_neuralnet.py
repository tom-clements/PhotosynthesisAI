import gc
from collections import defaultdict
from typing import List


import numpy as np
from sklearn.neural_network import MLPRegressor
from PhotosynthesisAI import Game
from PhotosynthesisAI.AI.RL.components.ML.neural_network import NNRegressor, ActFns
from PhotosynthesisAI.AI.RL.components.base.base_estimator import Estimator
from PhotosynthesisAI.AI.RL.components.environment.environment import WinRewardEnvironment
from PhotosynthesisAI.AI.RL.components.policy.epislon_greedy import make_epsilon_greedy_policy
from PhotosynthesisAI.AI.base import BaseAI
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.utils.utils import time_function

from logging import getLogger
logger = getLogger(__name__)


class SKlearnNNEstimator(Estimator):

    def _initialize_model(self, total_num_actions: int, start_features):
        model = MLPRegressor(hidden_layer_sizes=(128, 128,), learning_rate='constant')
        model.partial_fit(self._features_to_model_input(start_features), np.zeros((1, total_num_actions)))
        self.model = model

    @time_function
    def _features_to_model_input(self, state_features):
        return [state_features]

    def update(self, a, features, y):
        self.replay['features'].append(features)
        self.replay['y'].append(y)
        self.current_replay_size += 1
        if self.current_replay_size == self.replay_length:
            features = self.replay['features']
            y = self.replay['y']
            for i in range(self.replay_n_iter):
                self.model.partial_fit(features, y)
            self.current_replay_size = 0
            self.replay = dict(features=[], y=[])
        return


class SKlearnNNAI(BaseAI):

    def __init__(self, epsilon: float = 0.1, load_model=False):
        super().__init__()
        self.epsilon = epsilon
        self.policy = None
        self.estimator = None
        self.rewards = []
        self.discount_factor = 1
        self.load_model = load_model

    def initialise(self, game: Game):
        self._set_estimator(game)
        self._set_policy(game)
        self._set_environment(game)
        self.rewards = []
        if self.load_model:
            self.estimator.load_models()

    def _set_estimator(self, game: Game):
        if not self.estimator:
            self.estimator = SKlearnNNEstimator(game.total_num_actions, game.get_nn_features(self))

    def _set_environment(self, game: Game):
        self.env = WinRewardEnvironment(game=game, player=self)

    def _set_policy(self, game: Game):
        if not self.policy:
            self.policy = make_epsilon_greedy_policy(
                self.estimator, self.epsilon, game.total_num_actions)

    def play_turn(self, game):
        moves = self.starting_moves(game.board) if game.board.round_number in [0, 1] else self.moves_available(game.board)
        self.play_move(game, moves)

    @staticmethod
    @time_function
    def _action_to_move(action: int, available_moves: List[Move]) -> Move:
        move = [move for move in available_moves if move.move_id == action][0]
        return move

    @time_function
    def _get_policy_action(self, game, start_state, available_actions):
        probabilities = self.policy(start_state, available_actions=available_actions)
        action = np.random.choice(
            game.total_num_actions,
            p=probabilities
        )

        return action

    @time_function
    def get_td_target(self, new_state, reward, available_actions):
        # The Q-learning target policy is a greedy one, hence the `max`
        q_values_new_state = self.estimator.predict(state_features=new_state)

        # set value of invalid actions to be 0
        mask = np.ones(len(q_values_new_state), np.bool)
        mask[available_actions] = 0
        q_values_new_state[mask] = 0

        td_target = reward + self.discount_factor * np.nanmax(q_values_new_state)
        return td_target

    @time_function
    def play_move(self, game: Game, moves: List[Move]):
        available_actions = [move.move_id for move in moves]
        start_state = game.get_nn_features(self)
        action = self._get_policy_action(game, start_state, available_actions)
        try:
            move = self._action_to_move(action, moves)
        except:
            a = 5
        # Perform the action -> Get the reward and observe the next state
        new_state, reward = self.env.step(move)
        td_target = self.get_td_target(new_state, reward, available_actions)
        self.estimator.update(action, start_state, td_target)

    @time_function
    def save_model(self):
        self.estimator.pickle_models()

    @time_function
    def load_model(self):
        self.estimator.load_models()
