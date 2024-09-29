from collections import deque
import random
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    index = node.state.index(0)
    possible_moves = []
    if index % 3 > 0:
        possible_moves.append(-1)
    if index % 3 < 2:
        possible_moves.append(1)
    if index // 3 > 0:
        possible_moves.append(-3)
    if index // 3 < 2:
        possible_moves.append(3)
    
    for move in possible_moves:
        im = index + move
        new_state = list(node.state)
        new_state[index], new_state[im] = new_state[im], new_state[index]
        successor = Node(new_state, node)
        successors.append(successor)
    return successors


def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0
    while queue:
        node = queue.popleft()
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        nodes_explored = nodes_explored + 1
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1],nodes_explored
        for successor in get_successors(node):
            queue.append(successor)
    print('Total nodes explored', nodes_explored)
    return None

start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
s_node = Node(start_state)
D = 20
d = 0
while d <= D:
    goal_state = random.choice(list(get_successors(s_node))).state
    s_node = Node(goal_state)
    d = d+1
    print(goal_state)

solution,explored = bfs(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
    print('Total nodes explored', explored)
    print("Number Of Steps", len(solution))
else:
    print("No solution found.")
