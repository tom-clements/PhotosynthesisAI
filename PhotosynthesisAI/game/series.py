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
    def __init__(self, players: List[Player], num_matches: int):
        self.players = players
        self.num_matches = num_matches
        self.match_scores = {i + 1: [] for i in range(len(players))}
        self.game_states = []
        self.len_states = [0]
        self.len_duplicate_states = []

    @time_function
    def play(self, verbose: bool = False):
        for match_number in tqdm(range(self.num_matches)):
            # logger.info(f"playing match mumber {match_number}")
            game = Game(self.players)
            game.play(verbose=verbose)
            winners = game.get_winner()
            winner_numbers = [player.number for player in winners]
            losers = [player for player in self.players if player.number not in winner_numbers]
            for winner in winners:
                self.match_scores[winner.number].append(WIN_POINTS)
            for loser in losers:
                self.match_scores[loser.number].append(LOSE_POINTS)
            self.game = game
            self.game_states += game.states
            if match_number % 10 == 0:
                self.store_results()
            #     num_states = len(self.game_states)
            #     self.game_states = list(set(self.game_states))
            #     self.len_duplicate_states.append(num_states - len(self.game_states))
            #     self.len_states.append(len(self.game_states))
            #     if match_number != 0:
            #         states_per_match_added = (self.len_states[-1] - self.len_states[-2])/10
            #         logger.info(f"Match: {match_number}, "
            #                     f"Num unique states: {self.len_states[-1]}, "
            #                     f"Num duplicate states seen: {sum(self.len_duplicate_states[-1:])}, "
            #                     f"States per match added:{states_per_match_added}")

    def _get_results_df(self):
        results_df = pd.DataFrame(self.match_scores)
        results_df.columns = [f"{p.__class__.__name__}{p.number}" for p in self.players]
        for col in results_df:
            results_df[f"{col}_score"] = results_df[col].cumsum()
        results_df['win_rate'] = results_df[results_df.columns[0]].rolling(20).mean()
        return results_df

    @time_function
    def store_results(self):
        results_df = self._get_results_df()
        results_df.to_csv('results.csv', index=False)
        for player in self.players:
            player.save_progress()

    @classmethod
    def load_results_plot(cls):
        results_df = pd.read_csv('results.csv')
        cls.plot_results(results_df)

    def display_results(self):
        results_df = self._get_results_df()
        self.plot_results(results_df)

    @staticmethod
    def plot_results(results_df):
        plotting_columns = [col for col in results_df if 'score' in col] + ['win_rate']
        results_df[plotting_columns].plot(secondary_y='win_rate')
        plt.show()

    def get_function_time_metrics(self):
        df = pd.DataFrame(FUNCTION_TIMINGS).T.sort_values("time", ascending=False)
        df["%time"] = (100 * df.time) / df.loc["play"].time
        return df
