from PhotosynthesisAI.players import Human, RandomAI, DeepAI, NeuralNetworkAI, LinearAI, SKlearnNNAI
from PhotosynthesisAI.game import Series

if __name__ == "__main__":
    players = [DeepAI(name='deep'),  RandomAI()]
    series = Series(players=players, num_matches=100)
    series.play()
    series.display_results()
