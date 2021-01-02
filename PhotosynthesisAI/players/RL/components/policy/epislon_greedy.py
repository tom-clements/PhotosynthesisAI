from typing import List

import numpy as np

from PhotosynthesisAI.game.utils.utils import time_function


def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
    @time_function
    def policy_fn(state_features: List, available_actions: List):
        A = np.ones(num_actions, dtype=float) * epsilon / (len(available_actions))
        A = np.array([(val if i in available_actions else 0) for i, val in enumerate(A)])
        q_values = estimator.predict(state_features)

        # set value of invalid actions to be np.nan
        mask = np.ones(len(q_values), np.bool)
        mask[available_actions] = 0
        q_values[mask] = np.nan

        best_action = np.nanargmax(q_values)
        A[best_action] += 1.0 - epsilon
        return A

    return policy_fn
