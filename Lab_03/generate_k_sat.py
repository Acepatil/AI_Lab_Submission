import random

def generate_problem(k, m, n):
    # Initialize choices list for variables and their negations
    choices = []
    
    # Fill the choices array with variables (a, b, ..., z) and their negations (~a, ~b, ...)
    for i in range(n):
        var = chr(ord('a') + i)  # Get variable 'a', 'b', etc.
        choices.append(var)       # Add the positive literal
        choices.append(f"~{var}") # Add the negated literal

    ans = ""

    # Loop through m clauses
    for i in range(m):
        ans += "("
        clause = []

        # Loop through k literals in each clause
        for j in range(k):
            ind = random.randint(0, 2 * n - 1)  # Randomly choose a variable or its negation
            clause.append(choices[ind])         # Add it to the current clause

        ans += " | ".join(clause)  # Join the clause literals with OR
        ans += ")"
        
        if i != m - 1:
            ans += " & "  # Add AND between clauses
    
    print(ans)

# Input parameters
k = int(input("Enter the value of k (clause length): "))
m = int(input("Enter the number of clauses (m): "))
n = int(input("Enter the number of variables (n): "))

# Generate the logical formula
generate_problem(k, m, n)
