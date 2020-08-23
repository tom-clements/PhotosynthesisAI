from PhotosynthesisAI.game import Board
from PhotosynthesisAI.AI import RandomAI
from PhotosynthesisAI import Game

if __name__ == "__main__":
    players = [RandomAI(1), RandomAI(2)]
    game = Game(players)
    game.play()
    game.show()
