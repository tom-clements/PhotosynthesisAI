from PhotosynthesisAI.players import Human, RandomAI, ExpertSystemAI, NeuralNetworkAI, LinearAI, SKlearnNNAI
from PhotosynthesisAI.game import Series

if __name__ == "__main__":
    players = [SKlearnNNAI(load_model=True, name="best_one"), Human()]
    series = Series(players=players, num_matches=100)
    series.play()
    series.display_results()
