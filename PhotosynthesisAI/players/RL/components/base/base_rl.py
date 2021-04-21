from abc import abstractmethod
from typing import List

import numpy as np

from PhotosynthesisAI import Game
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.players.RL.components.environment.environment import WinRewardEnvironment
from PhotosynthesisAI.players.RL.components.policy.epislon_greedy import make_epsilon_greedy_policy
from PhotosynthesisAI.game.utils.utils import time_function
from PhotosynthesisAI.players.base import BaseAI


class BaseRL(BaseAI):
    def __init__(self, name: str):
        super().__init__(name)

    def initialise(self, game: Game):
        self._set_estimator(game)
        self._set_policy(game)
        self._set_environment(game)
        self.rewards = []

    def set_estimator(self, game: Game):
        pass

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
