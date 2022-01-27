import numpy as np
import random
from Helper_codes.graphics import *
from Helper_codes.question3 import Cell
from Helper_codes.question3 import IntPair
from Helper_codes.question3 import Player
from Helper_codes.question3 import NaivePlayer
from Helper_codes.question3 import Board
from Helper_codes.question3 import Game
import time
import matplotlib.pyplot as plt

# to do
class MinimaxPlayer(Player):
    start_time = None
    depth = 5

    def __init__(self, col, x, y):
        super().__init__(col, x, y)

    @staticmethod
    def canMove(player, board):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == j == 0:
                    continue
                if MinimaxPlayer.can_move_to(player, board, i, j):
                    return True
        return False

    def get_other_player(self, board):
        t1, t2 = board.players
        if t1.getCol() == self.getCol():
            return t2
        else:
            return t1

    def random_move(self, moves):
        if moves:
            return random.choice(moves)
        else:
            return None

    @staticmethod
    def get_priority_moves(player, board, moves):
        empty_cells = []
        yellow_cells = []
        for i, j in moves:
            if board.getSize() > player.getX() + i >= 0 and 0 <= player.getY() + j < board.getSize():
                cell = board.getCell(player.getX() + i, player.getY() + j)
                col = cell.getColor()
                if cell.getId() != -1:
                    yellow_cells.append((i, j))
                elif col == 0:
                    empty_cells.append((i, j))
        if player.getBuildingBlocks():
            return empty_cells, yellow_cells
        else:
            return yellow_cells, empty_cells

    @staticmethod
    def return_value(score, place, return_dest):
        return place if return_dest else score

    @staticmethod
    def can_move_to(player, board, i, j):
        return board.getSize() > player.getX() + i >= 0 and 0 <= player.getY() + j < board.getSize() and board.getCell(player.getX() + i, player.getY() + j).getColor() in [0, 3]

    def minValue(self, board, alpha, beta, depth):
        p2 = self.get_other_player(board)
        # if depth == 0 or not MinimaxPlayer.canMove(p2, board) or board.getNumberOfMoves() == board.maxNumberOfMoves:
        if depth == 0:
            return board.getScore(self.getCol())
        v = np.inf
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        random.shuffle(moves)
        for i, j in moves:
            if MinimaxPlayer.can_move_to(p2, board, i, j):
                new_board = Board(board)
                new_board.players = board.players
                new_board.move(IntPair(p2.getX() + i, p2.getY() + j), p2.getCol())
                v = min(v, self.maxValue(new_board, alpha, beta, depth - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        # if v == np.inf:
        #     return board.getScore(self.getCol())
        # else:
        return v

    def maxValue(self, board, alpha, beta, depth, dest = False):
        # if depth == 0 or not MinimaxPlayer.canMove(self, board) or board.getNumberOfMoves() == board.maxNumberOfMoves:
        if depth == 0:
            return MinimaxPlayer.return_value(board.getScore(self.getCol()), (self.getX(), self.getY()), dest)
        v = -np.inf
        best_move = None
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        random.shuffle(moves)
        moves1, moves2 = MinimaxPlayer.get_priority_moves(self, board, moves)
        for i, j in moves1 + moves2:
            if self.can_move_to(self, board, i, j):
                new_board = Board(board)
                new_board.move(IntPair(self.getX() + i, self.getY() + j), self.getCol())
                val = self.minValue(new_board, alpha, beta, depth - 1)
                if val > v:
                    v = val
                    best_move = (self.getX() + i, self.getY() + j)
                if v >= beta:
                    return MinimaxPlayer.return_value(v, best_move, dest)
                alpha = max(alpha, v)
        # if v == -np.inf:
        #     return MinimaxPlayer.return_value(board.getScore(self.getCol()), (self.getX(), self.getY()), dest)
        # else:
        return MinimaxPlayer.return_value(v, best_move, dest)

    def getMove(self, board):
        alpha = float('-inf')
        beta = float('inf')
        next = IntPair(-20, -20)

        if (board.getNumberOfMoves() == board.maxNumberOfMoves):
            return IntPair(-20, -20)

        if not (MinimaxPlayer.canMove(self, board)):
            return IntPair(-10, -10)

        MinimaxPlayer.start_time = time.time()

        x, y = self.maxValue(board, -np.inf, np.inf, MinimaxPlayer.depth, True)
        return IntPair(x, y)


################################################################
# p1 must be replace with minimaxPlayer                        #
################################################################
p1 = MinimaxPlayer(1, 0, 0)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
MinimaxPlayer.depth = 5
score1, score2 = g.start(numberOfMatches)
print(score1 / numberOfMatches)
