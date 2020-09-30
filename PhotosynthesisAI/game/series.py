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
        self.match_scores = {p.number: [] for p in players}
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
            # if match_number % 10 == 0:
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

    def display_results(self):
        results_df = pd.DataFrame(self.match_scores)
        plotting_columns = []
        for p in self.players:
            plotting_columns.append(f"{p.number}_score")
            results_df[f"{p.number}_score"] = results_df[p.number].cumsum()
        results_df[plotting_columns].plot()
        plt.show()

    def get_function_time_metrics(self):
        df = pd.DataFrame(FUNCTION_TIMINGS).T.sort_values("time", ascending=False)
        df["%time"] = (100 * df.time) / df.loc["play"].time
        return df
