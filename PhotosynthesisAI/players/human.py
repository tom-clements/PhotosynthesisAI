import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.player.moves import Move
from .. import Game


class Human(BaseAI):

    def _input_move(self, moves, options, text):
        print('Options', text, ', '.join(set(options)))
        picked_move_name = input('Pick a type of move to perform:')
        if picked_move_name == 'undo':
            picked_moves = None
        elif picked_move_name not in options:
            print('Invalid move selection, did you make a typo?')
            picked_move_name = self._input_move(moves, options, text)
        return picked_move_name

    def play_move(self, game: Game, moves: List[Move]):
        options = [move.get_name() for move in moves]
        picked_option = self._input_move(moves, options, 'pick a class of move: ')
        picked_moves = [move for move in moves if move.get_name() == picked_option]
        if picked_option == 'Buy':
            options = [move.tree.size for move in picked_moves]
            picked_option = self._input_move(moves, options, 'pick a size to buy: ')
            picked_moves = [move for move in picked_moves if move.tree.size == picked_option]
        elif picked_option == 'Collect':
            options = [move.tile.notation for move in picked_moves]
            picked_option = self._input_move(moves, options, 'pick a tile location to collect: ')
            picked_moves = [move for move in picked_moves if move.tile.notation == picked_option]
        elif picked_option == 'Grow':
            options = [move.tile.notation for move in picked_moves]
            picked_option = self._input_move(moves, options, 'pick a tile location to grow: ')
            picked_moves = [move for move in picked_moves if move.tile.notation == picked_option]
        elif picked_option == 'Plant':
            options = [move.tile.notation for move in picked_moves]
            picked_option = self._input_move(moves, options, 'pick a tile location to plant: ')
            picked_moves = [move for move in picked_moves if move.tile.notation == picked_option]
        if picked_option == 'undo':
            self.play_move(game, moves)
        else:
            game.execute_move(picked_moves[0])
