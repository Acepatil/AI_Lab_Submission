import numpy as np
import math

class Node:
    def __init__(self, parent, state, g_n, h_n, move=None):
        self.parent = parent
        self.state = state
        self.move = move
        self.g_n = g_n  # Path cost (g(n))
        self.h_n = h_n  # Heuristic cost (h(n))
        self.cost = g_n + h_n  # f(n) = g(n) + h(n)

    def __hash__(self):
        return hash(str(self.state.flatten()))

    def __eq__(self, other):
        return hash(''.join(self.state.flatten())) == hash(''.join(other.state.flatten())) 
    
    def __ne__(self, other):
        return hash(''.join(self.state.flatten())) != hash(''.join(other.state.flatten()))

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.hashes = {}

    def push(self, node):
        if hash(node) not in self.hashes:
            self.hashes[hash(node)] = 1
            self.queue.append(node)

    def pop(self):
        next_state = None
        state_cost = float('inf')
        index = -1

        for i in range(len(self.queue)):
            if self.queue[i].cost < state_cost:
                state_cost = self.queue[i].cost
                index = i

        return self.queue.pop(index)

    def is_empty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

class Environment:
    def __init__(self, start_state=None, goal_state=None):
        self.actions = [1, -1, 2, -2]  # 1: Up, -1: Down, 2: Right, -2: Left
        self.start_state = start_state if start_state is not None else self.initialize_initial_state()
        self.goal_state = goal_state if goal_state is not None else self.initialize_goal_state()

    def initialize_initial_state(self):
        start = np.array([[-1., -1.,  1.,  1.,  1., -1., -1.],
                  [-1., -1.,  1.,  1.,  1., -1., -1.],
                  [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
                  [-1., -1.,  1.,  1.,  1., -1., -1.],
                  [-1., -1.,  1.,  1.,  1., -1., -1.]])
        return start

    def initialize_goal_state(self):
        goal = np.array([[-1., -1.,  0.,  0.,  0., -1., -1.],
                 [-1., -1.,  0.,  0.,  0., -1., -1.],
                 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [ 0.,  0.,  0.,  1.,  0.,  0.,  0.],
                 [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
                 [-1., -1.,  0.,  0.,  0., -1., -1.],
                 [-1., -1.,  0.,  0.,  0., -1., -1.]])
        return goal

    def initial_state(self):
        return self.start_state
    
    def final_state(self):
        return self.goal_state

    def get_next_states(self, state):
        new_states = []
        spaces = [(i, j) for i in range(7) for j in range(7) if state[i][j] == 0]

        for space in spaces:
            x, y = space
            # Move top to bottom
            if x > 1 and state[x - 1][y] == 1 and state[x - 2][y] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x - 2][y] = 0
                new_state[x - 1][y] = 0
                move = f'({x - 2}, {y}) -> ({x}, {y})'
                new_states.append((new_state, move))

            # Move bottom to top
            if x < 5 and state[x + 1][y] == 1 and state[x + 2][y] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x + 2][y] = 0
                new_state[x + 1][y] = 0
                move = f'({x + 2}, {y}) -> ({x}, {y})'
                new_states.append((new_state, move))

            # Move left to right
            if y > 1 and state[x][y - 1] == 1 and state[x][y - 2] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x][y - 2] = 0
                new_state[x][y - 1] = 0
                move = f'({x}, {y - 2}) -> ({x}, {y})'
                new_states.append((new_state, move))

            # Move right to left
            if y < 5 and state[x][y + 1] == 1 and state[x][y + 2] == 1:
                new_state = state.copy()
                new_state[x][y] = 1
                new_state[x][y + 2] = 0
                new_state[x][y + 1] = 0
                move = f'({x}, {y + 2}) -> ({x}, {y})'
                new_states.append((new_state, move))
        return new_states

    def reached_goal(self, state):
        return np.array_equal(state, self.goal_state)

    def print_state(self, state):
        print("\n".join([" ".join([str(int(cell)) for cell in row]) for row in state]))
        print("")

class AStarSearch:
    def __init__(self, env, heuristic):
        self.frontier = PriorityQueue()
        self.explored = dict()
        self.start_state = env.initial_state()
        self.goal_state = env.final_state()
        self.env = env
        self.goal_node = None
        self.heuristic = heuristic

    def run(self):
        init_node = Node(parent=None, state=self.start_state, g_n=0, h_n=self.heuristic(self.start_state))
        self.frontier.push(init_node)
        while not self.frontier.is_empty():
            curr_node = self.frontier.pop()
            if self.env.reached_goal(curr_node.state):
                print("Reached goal!")
                self.goal_node = curr_node
                break

            if hash(curr_node) in self.explored:
                continue
            self.explored[hash(curr_node)] = curr_node

            next_states = self.env.get_next_states(curr_node.state)
            for next_state, move in next_states:
                h_n = self.heuristic(next_state)
                node = Node(parent=curr_node, state=next_state, g_n=curr_node.g_n + 1, h_n=h_n, move=move)
                self.frontier.push(node)

    def print_nodes(self):
        node = self.goal_node
        steps = []
        while node:
            steps.append(node)
            node = node.parent

        for step, node in enumerate(reversed(steps), start=1):
            print(f"Step {step}: {node.move}")
            self.env.print_state(node.state)

# Heuristic 1: Manhattan distance
def heuristic_manhattan(state):
    goal = np.zeros((7, 7))
    goal[3][3] = 1
    dist = 0
    for i in range(7):
        for j in range(7):
            if state[i][j] == 1:
                dist += abs(i - 3) + abs(j - 3)
    return dist

# Heuristic 2: Euclidean distance
def heuristic_euclidean(state):
    goal = np.zeros((7, 7))
    goal[3][3] = 1
    dist = 0
    for i in range(7):
        for j in range(7):
            if state[i][j] == 1:
                dist += math.sqrt((i - 3)**2 + (j - 3)**2)
    return dist

env = Environment()

# Run A* Search with Manhattan Heuristic
print("Running A* with Manhattan Distance Heuristic")
a_star_manhattan = AStarSearch(env, heuristic_manhattan)
a_star_manhattan.run()
a_star_manhattan.print_nodes()
print("Number of states explored: ", len(a_star_manhattan.explored))

# Run A* Search with Euclidean Heuristic
print("Running A* with Euclidean Distance Heuristic")
a_star_euclidean = AStarSearch(env, heuristic_euclidean)
a_star_euclidean.run()
a_star_euclidean.print_nodes()
print("Number of states explored: ", len(a_star_euclidean.explored))    
