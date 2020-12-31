from PhotosynthesisAI.AI.RL.components.base.base_envionment import Environment


class WinRewardEnvironment(Environment):

    def _get_reward(self, start_player_stats):
        # define reward as 100 points for winning and -100 for losing
        if self.game.is_game_over():
            winners = self.game.get_winner()
            if self.player.number in [winner.number for winner in winners]:
                return 100
            else:
                return -100
        else:
            return 0
