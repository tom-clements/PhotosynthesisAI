import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PhotosynthesisAI import Game
from PhotosynthesisAI.players.RL.components.base.base_estimator import Estimator
from PhotosynthesisAI.players.RL.components.base.base_rl import BaseRL
from PhotosynthesisAI.game.utils.utils import time_function

from logging import getLogger

logger = getLogger(__name__)


class Net(nn.Module):
    def __init__(self, num_features, total_num_actions):
        super().__init__()
        self.num_features = num_features
        self.fc1 = nn.Linear(num_features, 1024)
        self.fc2 = nn.Linear(1024, total_num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, X):
        output = self(torch.Tensor(X).view(-1, self.num_features)).detach().numpy()
        return output


class TorchEstimator(Estimator):
    def _initialize_model(self, total_num_actions: int, start_features):
        model = Net(len(start_features), total_num_actions)
        model.zero_grad()
        loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.1)
        self.model = model

    @time_function
    def _features_to_model_input(self, state_features):
        return [state_features]

    def update(self, features, y):
        self.replay["features"].append(features)
        self.replay["y"].append(y)
        self.current_replay_size += 1
        if self.current_replay_size == self.replay_length:
            features_set = self.replay["features"]
            y = np.array(self.replay["y"])
            for i in range(self.replay_n_iter):
                self.model.zero_grad()
                output = self.model(torch.tensor(features_set).view(-1, len(features_set[0])).float())
                loss = F.mse_loss(output, torch.tensor(y))
                loss.backward()
                self.optimizer.step()
            self.current_replay_size = 0
            self.replay = dict(features=[], y=[])


class TorchAI(BaseRL):
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
            self.estimator = TorchEstimator(
                game.total_num_actions, game.get_nn_features(self), load_model=self.load_model, name=self.name
            )
