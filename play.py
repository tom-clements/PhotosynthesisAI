from PhotosynthesisAI.AI import RandomAI, ExpertSystemAI, ExpertSystemAI2, LinearAI
from PhotosynthesisAI.game import Series

if __name__ == "__main__":
    players = [RandomAI(), LinearAI(epsilon=0.1)]
    series = Series(players=players, num_matches=1000)
    series.play()
    print(series.get_function_time_metrics())
    series.display_results()

