from string import ascii_lowercase
import random
from itertools import combinations
import numpy as np

def get_input():
    while True:
        try:
            print("Enter the number of clauses (positive integer):")
            m = int(input())
            if m <= 0:
                raise ValueError("Number of clauses must be positive.")
            print("Enter the number of variables (positive integer):")
            n = int(input())
            if n <= 0:
                raise ValueError("Number of variables must be positive.")
            return m, n
        except ValueError as e:
            print(e)

def generate_choices(n):
    """Generate variable choices and their negations."""
    choices = []
    for i in range(n):
        var = chr(ord('a') + i)  # Get variable 'a', 'b', etc.
        choices.append(var)       # Add the positive literal
        choices.append(f"~{var}") # Add the negated literal
    return choices

def generate_problem(k, m, n):
    """Generate a random propositional logic problem."""
    if n < k:
        raise ValueError("Number of variables must be greater than or equal to k.")
    
    choices = generate_choices(n)
    ans = ""

    for i in range(m):
        ans += "("
        clause = random.sample(choices, k)  # Use sample to avoid duplicates
        ans += " | ".join(clause)  # Join the clause literals with OR
        ans += ")"
        
        if i != m - 1:
            ans += " & "  # Add AND between clauses
    
    return choices, ans

def parse_formula(formula):
    """Parse the generated formula into a list of clauses."""
    formula = formula.replace("(", "").replace(")", "")
    clauses = formula.split(" & ")
    return [clause.split(" | ") for clause in clauses]

def create_problem(m, k, n):
    """Create a problem using variable combinations."""
    positive_vars = list(ascii_lowercase[:n])
    negative_vars = [var.upper() for var in positive_vars]
    variables = positive_vars + negative_vars
    threshold = 1
    problems = []
    all_combs = list(combinations(variables, k))

    while len(problems) < threshold:
        c = random.sample(all_combs, m)
        if c not in problems:
            problems.append(list(c))
    
    if not problems:
        raise ValueError("No valid problems could be generated.")

    return variables, problems

def assignment(variables, n):
    """Assign random values to variables."""
    for_positive = list(np.random.choice(2, n))
    for_negative = [1 - i for i in for_positive]
    assign = for_positive + for_negative
    return dict(zip(variables, assign))

def solve(problem, assign):
    """Evaluate the problem given an assignment of variables."""
    return sum(any(assign[val] for val in sub) for sub in problem)

def heuristic_score_1(problem, assignment):
    """Heuristic function 1: Count the number of satisfied clauses."""
    return solve(problem, assignment)

def heuristic_score_2(problem, assignment):
    """Heuristic function 2: Count the number of satisfied literals."""
    total_score = 0
    for clause in problem:
        if any(assignment[val] for val in clause):
            total_score += 1  # Increment for each satisfied clause
    return total_score



def hill_climbing(problem, assignment: dict, parent_score: float, received_steps: int, step: int, heuristic) -> tuple:
    """Perform hill climbing algorithm to optimize variable assignment with a heuristic."""
    
    best_assignment = assignment.copy()
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())
    
    while True:
        improved = False
        max_score = parent_score
        max_assignment = best_assignment.copy()
        
        for i in range(len(assign_values)):
            step += 1
            current_assignment = best_assignment.copy()
            current_assignment[assign_keys[i]] = 1 - assign_values[i]  # Flip the value
            
            # Evaluate the score using the selected heuristic
            current_score = heuristic(problem, current_assignment)

            if current_score > max_score:
                received_steps = step
                max_score = current_score
                max_assignment = current_assignment.copy()
                improved = True  # Mark that we found an improvement

        if not improved:  # If no improvements were found, break the loop
            break
        
        # Update for the next iteration
        parent_score = max_score
        best_assignment = max_assignment.copy()

    return best_assignment, parent_score, f"{received_steps}/{step - len(assign_values)}"

def beam_search(problem, assignment: dict, beam_width: int, step_size: int, heuristic) -> tuple:
    """Perform beam search algorithm with a heuristic."""
    
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())

    # Initial evaluation of the score
    initial_score = heuristic(problem, assignment)
    if initial_score == len(problem):
        return assignment, f"{step_size}/{step_size}"

    while True:
        possible_assignments = []
        possible_scores = []

        for i in range(len(assign_values)):
            step_size += 1
            current_assignment = assignment.copy()
            current_assignment[assign_keys[i]] = 1 - assign_values[i]  # Flip the assignment
            score = heuristic(problem, current_assignment)
            possible_assignments.append(current_assignment)
            possible_scores.append(score)

        # Check if any assignment achieves the goal score
        if len(problem) in possible_scores:
            index = possible_scores.index(len(problem))
            return possible_assignments[index], f"{step_size}/{step_size}"

        # Select the top `beam_width` assignments based on scores
        selected_indices = np.argsort(possible_scores)[-beam_width:]
        selected_assignments = [possible_assignments[i] for i in selected_indices]

        # Update the assignment for the next iteration
        assignment = selected_assignments[0]  # Choose the best one for the next iteration


def variable_neighbour(problem, assignment: dict, beam_width: int, step: int, heuristic: int) -> tuple:
    """Perform variable neighbourhood search algorithm with different heuristics."""
    
    assign_values = list(assignment.values())
    assign_keys = list(assignment.keys())

    # Initial evaluation of the score using the selected heuristic
    initial_score = heuristic_score_1(problem, assignment) if heuristic == 1 else heuristic_score_2(problem, assignment)
    
    if initial_score == len(problem):
        return assignment, f"{step}/{step}", beam_width

    while True:
        possible_assignments = []
        possible_scores = []

        for i in range(len(assign_values)):
            step += 1
            current_assignment = assignment.copy()
            current_assignment[assign_keys[i]] = 1 - assign_values[i]  # Flip the assignment
            score = heuristic_score_1(problem, current_assignment) if heuristic == 1 else heuristic_score_2(problem, current_assignment)

            possible_assignments.append(current_assignment)
            possible_scores.append(score)

        # Check if any assignment achieves the goal score
        if len(problem) in possible_scores:
            index = possible_scores.index(len(problem))
            return possible_assignments[index], f"{step}/{step}", beam_width

        # Select the top `beam_width` assignments based on scores
        selected_indices = np.argsort(possible_scores)[-beam_width:]
        selected_assignments = [possible_assignments[i] for i in selected_indices]

        # Update assignment for the next iteration
        assignment = selected_assignments[0]  # Choose the best one for the next iteration

        # Check for convergence
        if all(score <= initial_score for score in possible_scores):
            break
        
        initial_score = max(possible_scores)  # Update the initial score to the best score found

    return assignment, f"{step}/{step}", beam_width

def main():
    k = 3
    m, n = get_input()
    
    # Handle special cases
    if m == 1 and n == 1:
        problems = [[["a"]]]  # One clause with one variable
        variables = ["a", "~a"]
    elif m > 1 and n == 1:
        problems = [[["a"]] * m]  # All clauses the same variable
        variables = ["a", "~a"]
    elif m > 2 ** n:
        raise ValueError("Number of clauses must be less than or equal to 2^n.")
    else:
        variables, problems = create_problem(m, k, n)
    
    results = []

    for i, problem in enumerate(problems, start=1):
        assign = assignment(variables, n)
        initial_score = solve(problem, assign)

        # Run hill climbing with both heuristics
        best_assign_h1, score_h1, hp_h1 = hill_climbing(problem, assign, initial_score, 1, 1, heuristic_score_1)
        best_assign_h2, score_h2, hp_h2 = hill_climbing(problem, assign, initial_score, 1, 1, heuristic_score_2)

        # Run beam search with both heuristics
        beam_assign_h1, b_h1_penetration = beam_search(problem, assign, 3, 1, heuristic_score_1)
        beam_assign_h2, b_h2_penetration = beam_search(problem, assign, 3, 1, heuristic_score_2)

        # Test variable neighbourhood search with different heuristics
        var_assign_heuristic_1, v_penetration_1, _ = variable_neighbour(problem, assign, 1, 1, heuristic=1)
        var_assign_heuristic_2, v_penetration_2, _ = variable_neighbour(problem, assign, 1, 1, heuristic=2)


        results.append((i, problem, best_assign_h1, score_h1, hp_h1, best_assign_h2, score_h2, hp_h2,
                        beam_assign_h1, b_h1_penetration, beam_assign_h2, b_h2_penetration,var_assign_heuristic_1, v_penetration_1, var_assign_heuristic_2, v_penetration_2))

        print(f'Problem {i}: {problem}')
        print(f'Best Assignment (Hill Climbing H1): {best_assign_h1}, Score: {score_h1}, Steps: {hp_h1}')
        print(f'Best Assignment (Hill Climbing H2): {best_assign_h2}, Score: {score_h2}, Steps: {hp_h2}')
        print(f'Best Assignment (Beam Search H1): {beam_assign_h1}, Penetration: {b_h1_penetration}')
        print(f'Best Assignment (Beam Search H2): {beam_assign_h2}, Penetration: {b_h2_penetration}')
        print(f'Best Assignment (Variable Neighbourhood H1): {var_assign_heuristic_1}, Penetration: {v_penetration_1}')
        print(f'Best Assignment (Variable Neighbourhood H2): {var_assign_heuristic_2}, Penetration: {v_penetration_2}')

if __name__ == "__main__":
    main()
