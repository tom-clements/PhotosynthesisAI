from PhotosynthesisAI.game import Board
from PhotosynthesisAI.game import Player

if __name__ == '__main__':
    board = Board(2)
    p1 = Player(1)
    p2 = Player(2)
    players = [p1, p2]
    for player in players:
        moves = player.starting_moves(board)
        player.move(moves[0])
    board.show()
    board.start_round(players)
    moves = p1.moves_available(board)
    print(moves)
