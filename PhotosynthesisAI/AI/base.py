from PhotosynthesisAI.game.player import Player


class BaseAI(Player):
    def play_turn(self, board):
        while self.go_active:
            moves = self.starting_moves(board) if board.round_number == 0 else self.moves_available(board)
            move = self.pick_move(board, moves)
            self.move(move)
            if board.round_number == 0:
                break
        return
