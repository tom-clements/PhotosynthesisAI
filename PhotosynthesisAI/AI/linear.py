import itertools
from collections import namedtuple
from copy import deepcopy

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import SGDRegressor
from typing import List

from PhotosynthesisAI import Game


import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from ..game import Player


class LinearEstimator:

    def __init__(self, total_num_actions, start_features):
        self.total_num_actions = total_num_actions
        self.models = []
        for _ in range(total_num_actions):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([start_features], [0])
            self.models.append(model)

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

    def update(self, a, features, y):
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, num_actions):

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
                return 0
        else:
            l_points_gained = start_player.l_points - self.player.l_points
            return l_points_gained

    def step(self, move):
        start_player = deepcopy(self.player)
        self.game.execute_move(move)
        reward = self._get_reward(start_player)
        new_state = self.game.get_linear_features()
        return new_state, reward




class LinearAI(BaseAI):

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.policy = None
        self.discount_factor = 1
        self.rewards = []

    def initialise(self, game: Game):
        self._set_estimator(game)
        self._set_policy(game)
        self._set_environment(game)
        self.rewards = []

    def _set_estimator(self, game: Game):
        self.estimator = LinearEstimator(game.total_num_actions, game.get_linear_features())

    def _set_environment(self, game: Game):
        self.env = Environment(game=game, player=self)

    def _set_policy(self, game: Game):
        self.policy = make_epsilon_greedy_policy(
            self.estimator, self.epsilon, game.total_num_actions)

    def play_turn(self, game):
        moves = self.starting_moves(game.board) if game.board.round_number in [0, 1] else self.moves_available(game.board)
        self.play_move(game, moves)

    def action_to_move(self, action:int, available_moves: List[Move]) -> Move:
        move = [move for move in available_moves if move.move_id == action][0]
        return move

    def play_move(self, game: Game, moves: List[Move]):
        available_actions = [move.move_id for move in moves]
        start_state = game.get_linear_features()
        action = np.random.choice(
            game.total_num_actions,
            p=self.policy(start_state, available_actions=available_actions)
        )
        move = self.action_to_move(action, moves)

        new_state, reward = self.env.step(move)

        new_action = np.random.choice(
            game.total_num_actions,
            p=self.policy(new_state, available_actions=available_actions)
        )

        q_values_new_state = self.estimator.predict(state_features=new_state, available_actions=available_actions)

        td_target = reward + self.discount_factor * np.nanmax(q_values_new_state)
        self.estimator.update(action, start_state, td_target)

