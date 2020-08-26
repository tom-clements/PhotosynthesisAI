from PhotosynthesisAI.AI import RandomAI, ExpertSystemAI, ExpertSystemAI2
from PhotosynthesisAI.game import Series

if __name__ == "__main__":
    players = [ExpertSystemAI(1), RandomAI(2)]
    series = Series(players=players, num_matches=200)
    series.play()
    series.display_results()

