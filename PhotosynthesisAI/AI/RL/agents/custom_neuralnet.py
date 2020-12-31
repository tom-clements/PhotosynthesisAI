import gc
from typing import List


import numpy as np
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


class NeuralNetworkEstimator(Estimator):

    def _initialize_model(self, total_num_actions:int, start_features):
        for a in range(total_num_actions):
            model = NNRegressor(layer_sizes=(128, 1), learning_rate=0.05, output_activation_function=ActFns.identity)
            model.fit(self._features_to_model_input(start_features), [[0]])
            self.models.append(model)

    def _features_to_model_input(self, state_features):
        return np.array([state_features]).T


class NeuralNetworkAI(BaseAI):

    def __init__(self, epsilon, load_model=False):
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
            self.estimator = NeuralNetworkEstimator(game.total_num_actions, game.get_nn_features(self))

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
    def action_to_move(action:int, available_moves: List[Move]) -> Move:
        move = [move for move in available_moves if move.move_id == action][0]
        return move

    @time_function
    def play_move(self, game: Game, moves: List[Move]):
        available_actions = [move.move_id for move in moves]
        start_state = game.get_nn_features(self)
        action = np.random.choice(
            game.total_num_actions,
            p=self.policy(start_state, available_actions=available_actions)
        )
        move = self.action_to_move(action, moves)
        # Perform the action -> Get the reward and observe the next state
        new_state, reward = self.env.step(move)
        action = np.random.choice(
            game.total_num_actions,
            p=self.policy(start_state, available_actions=available_actions)
        )
        # new_action = np.random.choice(
        #     game.total_num_actions,
        #     p=self.policy(new_state, available_actions=available_actions)
        # )
        q_values_new_state = self.estimator.predict(state_features=new_state, available_actions=available_actions)

        # value that we should have got
        # The Q-learning target policy is a greedy one, hence the `max`
        td_target = reward + self.discount_factor * np.nanmax(q_values_new_state)
        self.estimator.update(action, start_state, td_target)

    @time_function
    def save_model(self):
        self.estimator.pickle_models()

    @time_function
    def load_model(self):
        self.estimator.load_models()
