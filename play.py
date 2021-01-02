from PhotosynthesisAI.players import Human, RandomAI, ExpertSystemAI, NeuralNetworkAI, LinearAI, SKlearnNNAI
from PhotosynthesisAI.game import Series

if __name__ == "__main__":
    players = [SKlearnNNAI(load_model=True, name="best_one"), Human()]
    series = Series(players=players, num_matches=1)
    series.play(show_board_each_turn=True, verbose=True)
    series.display_results()
