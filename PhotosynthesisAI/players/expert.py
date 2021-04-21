import random
from typing import List

from .base import BaseAI
from PhotosynthesisAI.game.components import Board
from PhotosynthesisAI.game.player.moves import Move
from PhotosynthesisAI.game.player import moves


class ExpertSystemAI(BaseAI):
    def pick_move(self, board: Board, availablle_moves: List[Move]) -> Move:
        collecting_moves = [move for move in availablle_moves if type(Move) == moves.Collect]
        planting_moves = [move for move in availablle_moves if type(Move) == moves.Plant]
        growing_moves = [move for move in availablle_moves if type(Move) == moves.Plant]

        # if you can grow or plant, don't end go
        if bool(growing_moves) | bool(planting_moves):
            availablle_moves = [move for move in availablle_moves if type(move) != moves.EndGo]

        # if near the end of the game, collect
        if (board.round_number > 15) & bool(collecting_moves):
            return random.choice(collecting_moves)

        # if near the start of the game, don't collect
        if (board.round_number < 10) & bool(collecting_moves):
            availablle_moves = [move for move in availablle_moves if type(move) != moves.Collect]

        # if have several planting moves, one with no trees nearby
        if len(planting_moves) > 1:
            # get number of surrounding trees
            surrounding_tiles = [
                sum([bool(tile.tree) for tile in board.get_surrounding_tiles(tuple(move.tile.coords), 1)])
                for move in planting_moves
            ]
            planting_move = planting_moves[surrounding_tiles.index(min(surrounding_tiles))]
            availablle_moves = [
                move for move in availablle_moves if not ((type(move) == moves.Plant) & (move != planting_move))
            ]

        return random.choice(availablle_moves)


class ExpertSystemAI2(BaseAI):
    def pick_move(self, board: Board, availablle_moves: List[Move]) -> Move:
        collecting_moves = [move for move in availablle_moves if type(Move) == moves.Collect]
        planting_moves = [move for move in availablle_moves if type(Move) == moves.Plant]
        growing_moves = [move for move in availablle_moves if type(Move) == moves.Plant]

        # if you can grow or plant, don't end go
        if bool(growing_moves) | bool(planting_moves):
            availablle_moves = [move for move in availablle_moves if type(move) != moves.EndGo]

        # if near the end of the game, collect
        if (board.round_number > 15) & bool(collecting_moves):
            return random.choice(collecting_moves)

        # if near the start of the game, don't collect
        if (board.round_number < 10) & bool(collecting_moves):
            availablle_moves = [move for move in availablle_moves if type(move) != moves.Collect]

        # if have several planting moves, one with no trees nearby
        if len(planting_moves) > 1:
            # get number of surrounding trees
            surrounding_tiles = [
                sum([bool(tile.tree) for tile in board.get_surrounding_tiles(tuple(move.tile.coords), 1)])
                for move in planting_moves
            ]
            planting_move = planting_moves[surrounding_tiles.index(min(surrounding_tiles))]
            availablle_moves = [
                move for move in availablle_moves if not ((type(move) == moves.Plant) & (move != planting_move))
            ]

        return random.choice(availablle_moves)
