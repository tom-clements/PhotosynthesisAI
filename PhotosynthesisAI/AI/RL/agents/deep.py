# import os
# import pickle
# from copy import deepcopy
# from typing import List
#
# import tensorflow as tf
#
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.agents import Sequential
# import numpy as np
# from sklearn.linear_model import SGDRegressor
# import PhotosynthesisAI
# from PhotosynthesisAI import Game
# from PhotosynthesisAI.AI.base import BaseAI
# from PhotosynthesisAI.game.player.moves import Move
# from PhotosynthesisAI.game import Player
# from PhotosynthesisAI.game.utils.utils import time_function
#
#
# class NNEstimator:
#
#     def __init__(self, total_num_actions, start_features):
#         self.total_num_actions = total_num_actions
#         self.agents = []
#         self.memory = {}
#     # @staticmethod
#     # def _get_default_path():
#     #     return os.path.join(os.path.dirname(PhotosynthesisAI.__file__), 'AI', 'RL', 'saved_models',
#     #                         'saved_models/linear.pkl')
#     #
#     # @time_function
#     # def pickle_models(self, path=None):
#     #     path = path if path else self._get_default_path()
#     #     with open(path, 'wb') as f:
#     #         pickle.dump(self.agents, f)
#     #
#     # @time_function
#     # def load_models(self, path=None):
#     #     path = path if path else self._get_default_path()
#     #     with open(path, 'rb') as f:
#     #         self.agents = pickle.load(f)
#
#     def predict(self, s, available_actions):
#         """
#         Predicts action values.
#
#         Args:
#           sess: Tensorflow session
#           s: State input of shape [batch_size, 4, 160, 160, 3]
#
#         Returns:
#           Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
#           action values.
#         """
#         return self.sess.run(self.predictions, { self.X_pl: s })
#
#     def update(self, s, a, y):
#         """
#         Updates the estimator towards the given targets.
#
#         Args:
#           sess: Tensorflow session object
#           s: State input of shape [batch_size, 4, 160, 160, 3]
#           a: Chosen actions of shape [batch_size]
#           y: Targets of shape [batch_size]
#
#         Returns:
#           The calculated loss on the batch.
#         """
#         feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
#         summaries, global_step, _, loss = self.sess.run(
#             [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
#             feed_dict)
#         if self.summary_writer:
#             self.summary_writer.add_summary(summaries, global_step)
#         return loss
#
#     def _build_model(self):
#         """
#         Builds the Tensorflow graph.
#         """
#
#         # Placeholders for our input
#         # Our input are 4 RGB frames of shape 160, 160 each
#         self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
#         # The TD target value
#         self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
#         # Integer id of which action was selected
#         self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
#
#         X = tf.to_float(self.X_pl) / 255.0
#         batch_size = tf.shape(self.X_pl)[0]
#
#         # Three convolutional layers
#         conv1 = tf.contrib.layers.conv2d(
#             X, 32, 8, 4, activation_fn=tf.nn.relu)
#         conv2 = tf.contrib.layers.conv2d(
#             conv1, 64, 4, 2, activation_fn=tf.nn.relu)
#         conv3 = tf.contrib.layers.conv2d(
#             conv2, 64, 3, 1, activation_fn=tf.nn.relu)
#
#         # Fully connected layers
#         flattened = tf.contrib.layers.flatten(conv3)
#         fc1 = tf.contrib.layers.fully_connected(flattened, 512)
#         self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))
#
#         # Get the predictions for the chosen actions only
#         gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
#         self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)
#
#         # Calcualte the loss
#         self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
#         self.loss = tf.reduce_mean(self.losses)
#
#         # Optimizer Parameters from original paper
#         self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
#         self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
#
#         # Summaries for Tensorboard
#         self.summaries = tf.summary.merge([
#             tf.summary.scalar("loss", self.loss),
#             tf.summary.histogram("loss_hist", self.losses),
#             tf.summary.histogram("q_values_hist", self.predictions),
#             tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
#         ])
#
#
# def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
#
#     @time_function
#     def policy_fn(state_features: List, available_actions: List):
#         A = np.ones(num_actions, dtype=float) * epsilon / (len(available_actions))
#         A = np.array([(val if i in available_actions else 0) for i, val in enumerate(A)])
#         q_values = estimator.predict(state_features, available_actions=available_actions)
#         best_action = np.nanargmax(q_values)
#         A[best_action] += (1.0 - epsilon)
#         return A
#
#     return policy_fn
#
#
#
# class DeepAI(BaseAI):
#
#     def __init__(self, epsilon, load_model=False):
#         super().__init__()
#         self.epsilon = epsilon
#         self.policy = None
#         self.estimator = None
#         self.rewards = []
#         self.discount_factor = 1
#         self.load_model = load_model
#
#     def initialise(self, game: Game):
#         self._set_estimator(game)
#         self._set_policy(game)
#         self._set_environment(game)
#         self.rewards = []
#         if self.load_model:
#             self.estimator.load_models()
#
#     def _set_estimator(self, game: Game):
#         if not self.estimator:
#             self.estimator = NNEstimator(game.total_num_actions, game.get_linear_features(self))
#
#     def _set_environment(self, game: Game):
#         self.env = Environment(game=game, player=self)
#
#     def _set_policy(self, game: Game):
#         if not self.policy:
#             self.policy = make_epsilon_greedy_policy(
#                 self.estimator, self.epsilon, game.total_num_actions)
#
#     # def pick_move(self, game: Game, available_moves: List[Move]) -> Move:
#     #     action = np.random.choice(game.total_num_actions, p=self.policy(game.get_linear_features()))
#     #     return
#
#     def play_turn(self, game):
#         moves = self.starting_moves(game.board) if game.board.round_number in [0, 1] else self.moves_available(game.board)
#         self.play_move(game, moves)
#
#     @time_function
#     def action_to_move(self, action:int, available_moves: List[Move]) -> Move:
#         move = [move for move in available_moves if move.move_id == action][0]
#         return move
#
#     @time_function
#     def play_move(self, game: Game, moves: List[Move]):
#         available_actions = [move.move_id for move in moves]
#         start_state = game.get_linear_features(self)
#         action = np.random.choice(
#             game.total_num_actions,
#             p=self.policy(start_state, available_actions=available_actions)
#         )
#         move = self.action_to_move(action, moves)
#         # Perform the action -> Get the reward and observe the next state
#         new_state, reward = self.env.step(move)
#         # new_action = np.random.choice(
#         #     game.total_num_actions,
#         #     p=self.policy(new_state, available_actions=available_actions)
#         # )
#
#         q_values_new_state = self.estimator.predict(state_features=new_state, available_actions=available_actions)
#
#         # value that we should have got
#         # The Q-learning target policy is a greedy one, hence the `max`
#         td_target = reward + self.discount_factor * np.nanmax(q_values_new_state)
#         self.estimator.update(action, start_state, td_target)
#
#     @time_function
#     def save_progress(self):
#         self.estimator.pickle_models()
