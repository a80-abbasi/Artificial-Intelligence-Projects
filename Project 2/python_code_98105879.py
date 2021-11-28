student_number = 98105879
Name = 'Ali'
Last_Name = 'Abbasi'

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def f_1(x):
    return (x * x * np.cos(x / 10) - x) / 100

def f_2(x):
    return np.log(np.sqrt(np.sin(x / 20)))

def f_3(x):
    return np.log(np.cos(x) + 45 / x)

def draw(func, x_range):
    plt.plot(x_range, func(x_range))
    plt.show()

def gradiant_descent(func, initial_point: float, learning_rate: float, max_iterations: int):
    h = learning_rate / 20
    x = initial_point
    for i in range(max_iterations):
        derivative = (func(x + h) - func(x)) / h  # we use this approximation for derivative of func
        if derivative == 0:
            break
        x = x - learning_rate * derivative
    return x

def f(x_1, x_2):
    return 2 * x_1 * x_1 + 3 * x_2 * x_2 - 4 * x_1 * x_2 - 50 * x_1 + 6 * x_2

def gradiant_descent(func, initial_point: Tuple, learning_rate: float, threshold: float, max_iterations: int):
    x_1_sequence = [initial_point[0]]
    x_2_sequence = [initial_point[1]]
    x, y = initial_point
    for i in range(max_iterations):
        x, y = update_points(func, x, y, learning_rate)
        if x > threshold or y > threshold:
            break
        x_1_sequence.append(x)
        x_2_sequence.append(y)
    
    return x_1_sequence, x_2_sequence

def update_points(func, x_1, x_2, learning_rate):
    h = learning_rate / 20
    return x_1 - learning_rate * (func(x_1 + h, x_2) - func(x_1, x_2)) / h, x_2 - learning_rate * (func(x_1, x_2 + h) - func(x_1, x_2)) / h

result_sequences = []
for rate in learning_rates:
    result_sequences.append(gradiant_descent(f, initial_point, rate, threshold, max_iterations))

res1 = result_sequences[0]
draw_points_sequence(f, res1[0], res1[1])

res2 = result_sequences[1]
draw_points_sequence(f, res2[0], res2[1])

res3 = result_sequences[2]
draw_points_sequence(f, res3[0], res3[1])

res4 = result_sequences[3]
draw_points_sequence(f, res4[0], res4[1])
print(res4)



def ac_3():
    pass

def backtrack():
    pass

def backtracking_search():
    return backtrack()



class MinimaxPlayer(Player):
    
    def __init__(self, col, x, y):
        super().__init__(col, x, y)

    def minValue(self, board, alpha, beta, depth):
        pass
    
    def maxValue(self, board, alpha, beta, depth):
        pass
    
    def getMove(self, board):
        pass

p1 = NaivePlayer(1, 0, 0)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)









