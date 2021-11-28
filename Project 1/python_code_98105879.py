student_number = 98105879
Name = 'Ali'
Last_Name = 'Abbasi'

import heapq
from math import inf




def add_neighbors(roads, floristries, fringe, node):
    total_cost, costs, poses, flowers = node
    pos1, pos2 = poses
    cost1, cost2 = costs

    for dest, road_cost in roads[pos1]:
        heapq.heappush(fringe, make_state(cost1 + road_cost, cost2, dest, pos2, flowers.union(floristries[dest])))

    for dest, road_cost in roads[pos2]:
        heapq.heappush(fringe, make_state(cost1, cost2 + road_cost, pos1, dest, flowers.union(floristries[dest])))


def graph_search(roads, floristries, n, k):
    fringe = [make_state(0, 0, 1, 1, set(floristries[1]))]
    while True:
        if fringe:
            node = heapq.heappop(fringe)
            if goal_test(node, n, k):
                return node[0]

            add_neighbors(roads, floristries, fringe, node)
        else:
            return inf


def make_state(cost1, cost2, pos1, pos2, flowers):
    return max(cost2, cost1), (cost1, cost2), (pos1, pos2), flowers


def goal_test(state: tuple, n, k):
    total_cost, costs, poses, flowers = state
    pos1, pos2 = poses
    if pos1 == pos2 == n:
        return len(flowers) == k
    else:
        return False

def solve(N, M, K, NUMS, roads):
    floristries = [[]]  # todo
    for flower_list in NUMS:
        flower_list.pop(0)
        floristries.append(flower_list)

    graph: list = []
    for i in range(N + 1):
        graph.append([])
    for origin, dest, cost in roads:
        graph[origin].append((dest, cost))
        graph[dest].append((origin, cost))

    return graph_search(graph, floristries, N, K)

import time

def heur_displaced(state):
    return len(set(state.boxes.keys()).difference(state.storage.keys()))

def heur_manhattan_distance(state):
    heu = 0
    for i, box in enumerate(state.boxes):
        if state.restrictions and state.restrictions[i]:
            valid_storages = state.restrictions[i]
        else:
            valid_storages = state.storage
        min_dist = math.inf
        box_x, box_y = box
        for storage_x, storage_y in valid_storages:
            manhattan = abs(storage_x-box_x) + abs(storage_y-box_y)
            min_dist = min(min_dist, manhattan)
        heu += min_dist
    return heu

def heur_euclidean_distance(state):  
    heu = 0
    for i, box in enumerate(state.boxes):
        if state.restrictions and state.restrictions[i]:
            valid_storages = state.restrictions[i]
        else:
            valid_storages = state.storage
        min_dist = math.inf
        for storage_coordinate in valid_storages:
            manhattan = math.dist(box, storage_coordinate)
            min_dist = min(min_dist, manhattan)
        heu += min_dist
    return heu



def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    best_path_cost = float("inf")
    time_remain = 8
    iter = 0
    optimal_final = None
    cost_bound = math.inf, math.inf, math.inf

    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)

    while (time_remain > 0) and not se.open.empty():
        start_time = time.time()        
        final = se.search(timebound=10, costbound=cost_bound)
        if final:
            cost_bound = final.gval, 0, final.gval  # hval is zero in final states
            if final.gval < best_path_cost:
                best_path_cost = final.gval
                optimal_final = final
            end_time = time.time()
            time_remain -= end_time - start_time
        else:
            break
        
    try:
        return optimal_final
    except:
        return final

    return False





edge_count = np.count_nonzero(graph_matrix)
print(edge_count)

def random_state_generator(n):
    return np.array(np.random.choice(a=[False, True], size=n))

def neighbour_state_generator(state):
    new_state = state.copy()
    vertex_to_change = random.randint(0, len(new_state) - 1)
    previous_value = state[vertex_to_change]
    new_state[vertex_to_change] = ~previous_value
    return new_state, previous_value, vertex_to_change

def cost_function(graph_matrix, state, A=1, B=1):
    return A * sum(state) + B * np.count_nonzero((np.dot(~np.array([state]).T, ~np.array([state]))) * graph_matrix)

deg = [np.count_nonzero(x) / edge_count for x in graph_matrix]

def prob_accept(current_state, delta_cost, changed_vertex, t):
    term = 1 - deg[changed_vertex] if current_state[changed_vertex] else 1 + deg[changed_vertex]
    return exp(-delta_cost * term / t)

def accept(current_state, next_state, changed_vertex, t):
    p = prob_accept(current_state, next_state, changed_vertex, t)
    return np.random.choice([True, False], p=[p, 1 - p])

def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.show()

plot_cost(cost_list)

import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt
import math

def cost_function(graph_matrix, state, A=1, B=1):
    return A * sum(state) + B * np.count_nonzero((np.dot(~np.array([state]).T, ~np.array([state]))) * graph_matrix)

graph_matrix =[]
def load_data(path = "./Inputs/test-q3-q4.txt"):
    with  open(path , 'r') as f:
        lines = f.readlines()
        number_of_vertices = int(lines[0])
        for i in range(number_of_vertices):
            line_split = lines[i+1].split(',');
            graph_matrix.append([])
            for j in range(number_of_vertices):
                graph_matrix[i].append(int(line_split[j]))
load_data()

def calculate_costs(graph, population):
    costs = []
    for x in population:
        costs.append(cost_function2(graph, x))
    return costs

def population_generation(n, k): 
    return np.array(np.random.choice(a=[False, True], size=(k, n)))

def cost_function2(graph, state):
    return cost_function(graph,state, A=1, B=5)

def tournament_selection(graph, population, costs):
    middle = len(population) // 2
    for i in range(middle):
        if costs[middle + i] < costs[i]:
            population[i] = population[middle + i]
    return population

def crossover(graph, parent1, parent2):
    index = random.randint(0, len(parent1) - 1)
    child1 = np.concatenate([parent1[0:index + 1], parent2[index+1:]])
    child2 = np.concatenate([parent2[0:index + 1], parent1[index+1:]])
    return child1, child2

def mutation(graph, chromosme, probability):
    p = random.uniform(0, 1)
    if p > probability:
        index = random.randint(0, len(chromosme)-1)
        chromosme[index] = not chromosme[index]

def genetic_algorithm(graph_matrix, mutation_probability=0.1, pop_size=100, max_generation=100):
    population = population_generation(len(graph_matrix), pop_size)
    best_cost = math.inf
    best_solution = None
    middle = pop_size // 2
    costs = calculate_costs(graph_matrix, population)
    for i in range(max_generation):
        population = tournament_selection(graph_matrix, population, costs)

        for i in range(pop_size // 4):
            child1, child2 = crossover(graph_matrix, population[2 * i], population[2 * i + 1])
            population[middle + 2 * i] = child1
            population[middle + 2 * i + 1] = child2

        for i in range(pop_size):
            mutation(graph_matrix, population[i], mutation_probability)
        
        costs = calculate_costs(graph_matrix, population)
        for i in range(pop_size):
            if costs[i] < best_cost:
                best_solution = population[i]
                best_cost = costs[i]
    
    return best_cost, best_solution

