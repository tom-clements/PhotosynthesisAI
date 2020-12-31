from PhotosynthesisAI.AI import RandomAI, ExpertSystemAI, NeuralNetworkAI, LinearAI, SKlearnNNAI
from PhotosynthesisAI.game import Series


if __name__ == "__main__":
    players = [SKlearnNNAI(), SKlearnNNAI()]
    series = Series(players=players, num_matches=50000)
    series.play()
    players[0].save_progress()
    series.display_results()

