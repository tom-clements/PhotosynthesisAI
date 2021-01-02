import logging
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from PhotosynthesisAI.game.player import Player
from PhotosynthesisAI.game import Game
from PhotosynthesisAI.game.utils.utils import FUNCTION_TIMINGS, time_function

logger = logging.getLogger("Game")
logging.basicConfig(level=logging.INFO)


WIN_POINTS = 1
LOSE_POINTS = 0


class Series:
    def __init__(
        self,
        players: List[Player],
        num_matches: int,
    ):
        self.players = players
        self.num_matches = num_matches
        self.match_scores = []
        self.len_states = [0]
        self.len_duplicate_states = []
        self.most_recent_game = None

    @time_function
    def play(self, verbose: bool = False, show_board_each_turn: bool = False):
        for match_number in tqdm(range(self.num_matches)):
            if verbose:
                logger.info(f"playing match mumber {match_number}")
            game = Game(self.players)
            self.most_recent_game = game
            game.play(verbose=verbose, show_board_each_turn=show_board_each_turn)
            winners = game.get_winner()
            winners_numbers = [winner.number for winner in winners]
            losers = [player for player in self.players if player.number not in winners_numbers]
            result = {
                "match_number": match_number,
            }
            win_results = [
                {
                    f"{winner.name}_{winner.number}": WIN_POINTS,
                    f"{winner.name}_{winner.number}_color": winner.color,
                    f"{winner.name}_{winner.number}_score": winner.score,
                }
                for winner in winners
            ]
            loss_results = [
                {
                    f"{loser.name}_{loser.number}": LOSE_POINTS,
                    f"{loser.name}_{loser.number}_color": loser.color,
                    f"{loser.name}_{loser.number}_score": loser.score,
                }
                for loser in losers
            ]
            for r in win_results:
                result.update(r)
            for r in loss_results:
                result.update(r)
            self.match_scores.append(result)

    def get_results_df(self):
        results_df = pd.DataFrame(self.match_scores)
        for player in self.players:
            results_df[f"{player.name}_{player.number}_total"] = results_df[f"{player.name}_{player.number}"].cumsum()
        return results_df.set_index("match_number")

    @time_function
    def store_results(self):
        results_df = self.get_results_df()
        results_df.to_csv("results.csv", index=False)
        for player in self.players:
            player.save_progress()

    @classmethod
    def load_results_plot(cls):
        results_df = pd.read_csv("results.csv")
        cls.plot_results(results_df)

    def display_results(self):
        results_df = self.get_results_df()
        self.plot_results(results_df)

    def plot_results(self, results_df):
        plotting_columns = [f"{player.name}_{player.number}_total" for player in self.players]
        results_df[plotting_columns].plot()
        plt.show()

    def get_function_time_metrics(self):
        df = pd.DataFrame(FUNCTION_TIMINGS).T.sort_values("time", ascending=False)
        df["%time"] = (100 * df.time) / df.loc["play"].time
        return df
