import os
import pickle
from copy import deepcopy


import numpy as np
from sklearn.linear_model import SGDRegressor
from typing import List

import PhotosynthesisAI
from PhotosynthesisAI import Game
from PhotosynthesisAI.AI.base import BaseAI
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game import Player
from PhotosynthesisAI.game.utils.utils import time_function


class LinearEstimator:

    def __init__(self, total_num_actions, start_features):
        self.total_num_actions = total_num_actions
        self.models = []
        self.memory = {}
        for a in range(total_num_actions):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([start_features], [0])
            self.models.append(model)
            self.memory[a] = {'features': [], 'y': []}


    @time_function
    def predict(self, state_features: List, available_actions, action=None):
        if action is not None:
            prediction = self.models[action].predict([state_features])
            return prediction[0]

        else:
            predictions = np.array(
                [
                    self.models[i].predict([state_features]) if i in available_actions else [np.nan]
                    for i in range(self.total_num_actions)

                ]
            )
            return predictions.reshape(-1)

    def add_to_memory(self, a, features, y):
        self.memory[a]['features'].append(features)
        self.memory[a]['y'].append(y)

    @time_function
    def update_from_memory(self):
        for a in range(self.total_num_actions):
            if self.models[a]['features']:
                self.models[a].partial_fit(self.memory[a]['features'], self.memory[a]['y'])

    @time_function
    def update(self, a, features, y):
        self.models[a].partial_fit([features], [y])

    @staticmethod
    def _get_default_path():
        return os.path.join(os.path.dirname(PhotosynthesisAI.__file__), 'AI', 'RL', 'saved_models', 'linear.pkl')

    @time_function
    def pickle_models(self, path=None):
        path = path if path else self._get_default_path()
        with open(path, 'wb') as f:
            pickle.dump(self.models, f)

    @time_function
    def load_models(self, path=None):
        path = path if path else self._get_default_path()
        with open(path, 'rb') as f:
            self.models = pickle.load(f)


def make_epsilon_greedy_policy(estimator, epsilon, num_actions):

    @time_function
    def policy_fn(state_features: List, available_actions: List):
        A = np.ones(num_actions, dtype=float) * epsilon / (len(available_actions))
        A = np.array([(val if i in available_actions else 0) for i, val in enumerate(A)])
        q_values = estimator.predict(state_features, available_actions=available_actions)
        best_action = np.nanargmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


class Environment:

    def __init__(self, game: Game, player: Player):
        self.game = game
        self.player = player

    def action_to_move(self):
        return

    def _get_reward(self, start_player):
        # define reward as number of l_points earned
        # and 100 points for winning
        if self.game.is_game_over():
            winners = self.game.get_winner()
            if self.player.number in [winner.number for winner in winners]:
                return 100
            else:
                return -100
        else:
            l_points_gained = start_player.l_points - self.player.l_points
            score_gained = start_player.score - self.player.score
            return l_points_gained + score_gained

    def step(self, move):
        start_player = deepcopy(self.player)
        self.game.execute_move(move)
        reward = self._get_reward(start_player)
        new_state = self.game.get_linear_features(self.player)
        return new_state, reward





class LinearAI(BaseAI):

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
            self.estimator = LinearEstimator(game.total_num_actions, game.get_linear_features(self))

    def _set_environment(self, game: Game):
        self.env = Environment(game=game, player=self)

    def _set_policy(self, game: Game):
        if not self.policy:
            self.policy = make_epsilon_greedy_policy(
                self.estimator, self.epsilon, game.total_num_actions)

    def play_turn(self, game):
        moves = self.starting_moves(game.board) if game.board.round_number in [0, 1] else self.moves_available(game.board)
        self.play_move(game, moves)

    @time_function
    def action_to_move(self, action:int, available_moves: List[Move]) -> Move:
        move = [move for move in available_moves if move.move_id == action][0]
        return move

    @time_function
    def play_move(self, game: Game, moves: List[Move]):
        available_actions = [move.move_id for move in moves]
        start_state = game.get_linear_features(self)
        action = np.random.choice(
            game.total_num_actions,
            p=self.policy(start_state, available_actions=available_actions)
        )
        move = self.action_to_move(action, moves)
        # Perform the action -> Get the reward and observe the next state
        new_state, reward = self.env.step(move)
        # new_action = np.random.choice(
        #     game.total_num_actions,
        #     p=self.policy(new_state, available_actions=available_actions)
        # )

        q_values_new_state = self.estimator.predict(state_features=new_state, available_actions=available_actions)

        td_target = reward + self.discount_factor * np.nanmax(q_values_new_state)
        self.estimator.update(action, start_state, td_target)

    @time_function
    def save_progress(self):
        self.estimator.pickle_models()
