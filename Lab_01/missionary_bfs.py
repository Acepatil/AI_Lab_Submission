from collections import deque

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    successors = []
    value = 0
    index = node.state.index(value)
    moves = [-2, -1, 1, 2]
    for move in moves:
        im = index + move
        if im >= 0 and im <= 6:
            if move > 0 and node.state[im] == -1:
                new_state = list(node.state)
                temp = new_state[im]
                new_state[im] = new_state[index]
                new_state[index] = temp
                successor = Node(new_state, node)
                successors.append(successor) 
            elif move < 0 and node.state[im] == 1:
                new_state = list(node.state)
                temp = new_state[im]
                new_state[im] = new_state[index]
                new_state[index] = temp
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
        nodes_explored += 1
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

start_state = [1, 1, 1, 0, -1, -1, -1]
goal_state = [-1, -1, -1, 0, 1, 1, 1]
s_node = Node(start_state)

solution,explored = bfs(start_state, goal_state)
if solution:
    for step in solution:
        print(step)
    print('Total nodes explored', explored)
    print("Number Of Steps", len(solution))
else:
    print("No solution found.")
