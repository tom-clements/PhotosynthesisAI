import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from PhotosynthesisAI.game.player import Player
from PhotosynthesisAI.game import Game


logger = logging.getLogger("Game")
logging.basicConfig(level=logging.INFO)


WIN_POINTS = 1
LOSE_POINTS = 0


class Series:
    def __init__(self, players: List[Player], num_matches: int, verbose: bool = True):
        self.players = players
        self.num_matches = num_matches
        self.verbose = verbose
        self.match_scores = {p.number: [] for p in players}

    def play(self):
        for match_number in range(self.num_matches):
            logger.info(f"playing match mumber {match_number}")
            game = Game(self.players)
            game.play()
            winners = game.get_winner()
            winner_numbers = [player.number for player in winners]
            losers = [
                player
                for player in self.players
                if player.number not in winner_numbers
            ]
            for winner in winners:
                self.match_scores[winner.number].append(WIN_POINTS)
            for loser in losers:
                self.match_scores[loser.number].append(LOSE_POINTS)

    def display_results(self):
        results_df = pd.DataFrame(self.match_scores)
        plotting_columns = []
        for p in self.players:
            plotting_columns.append(f"{p.number}_score")
            results_df[f"{p.number}_score"] = results_df[p.number].cumsum()
        results_df[plotting_columns].plot()
        plt.show()


