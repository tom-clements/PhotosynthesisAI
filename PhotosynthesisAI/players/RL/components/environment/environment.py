from PhotosynthesisAI.players.RL.components.base.base_envionment import Environment


class WinRewardEnvironment(Environment):
    def _get_reward(self, start_player_stats):
        # return 0
        current_score = self.player.score
        current_points = sum([p for points in self.player.l_points_earned_history.values() for p in points])
        score_diff = current_score - start_player_stats["score"]
        l_diff = current_points - start_player_stats["points"]
        # l_diff = 0
        return score_diff + l_diff

    def get_reward_game(self):
        winners = self.game.get_winner()
        if self.player.number in [winner.number for winner in winners]:
            return 100
        else:
            return -100
