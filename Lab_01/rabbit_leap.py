def print_state(state):
    """Function to print the current state of the stones."""
    print(" ".join(str(x) for x in state))


def is_goal_state(state):
    """Check if the current state is the goal state."""
    return state == [0, 0, 0, -1, 1, 1, 1]


def get_successor(state):
    """Find all possible moves for the rabbits."""
    moves = []
    for i in range(len(state)):
        if state[i] == 1:  # East-bound rabbit
            # Move one step forward if possible
            if i + 1 < len(state) and state[i + 1] == -1:
                moves.append((i, i + 1))
            # Jump over one rabbit if possible
            if i + 2 < len(state) and state[i + 1] == 0 and state[i + 2] == -1:
                moves.append((i, i + 2))
        elif state[i] == 0:  # West-bound rabbit
            # Move one step forward if possible
            if i - 1 >= 0 and state[i - 1] == -1:
                moves.append((i, i - 1))
            # Jump over one rabbit if possible
            if i - 2 >= 0 and state[i - 1] == 1 and state[i - 2] == -1:
                moves.append((i, i - 2))
    return moves


def make_move(state, move):
    """Apply a move to the state."""
    new_state = state[:]
    new_state[move[1]] = new_state[move[0]]
    new_state[move[0]] = -1
    return new_state


def dfs(initial_state):
    """Solve the Rabbit Leap problem using a simple backtracking approach."""
    stack = [(initial_state, [])]  # (current state, path of moves)
    visited = set()

    while stack:
        current_state, path = stack.pop()

        # Check if we have already visited this state
        if tuple(current_state) in visited:
            continue

        visited.add(tuple(current_state))
        path = path + [current_state]

        # Check if the current state is the goal state
        if is_goal_state(current_state):
            return path

        # Find all possible moves from the current state
        moves = get_successor(current_state)
        for move in moves:
            new_state = make_move(list(current_state), move)
            stack.append((new_state, path))

    return None


# Initial state: 1, 1, 1 (East-bound rabbits), -1 (Empty), 0, 0, 0 (West-bound rabbits)
initial_state = [1, 1, 1, -1, 0, 0, 0]
path = dfs(initial_state)

if path:
    print("Solution Found:")
    for step in path:
        print_state(step)
    print("Number of moves:", len(path) - 1)
else:
    print("No solution found")
