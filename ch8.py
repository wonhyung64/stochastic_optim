#%%
import numpy as np
from copy import copy
from scipy.stats import bernoulli
## 8.4
# %%
def inverse(permutation):
    permutation_copy = copy(permutation)
    change_list = permutation_copy[1:7]
    changed_list = change_list[-1::-1]

    permutation_copy[1:7] = changed_list
    return permutation_copy

def translate(permutation):
    permutation_copy = copy(permutation)
    change_list = permutation_copy[1:8]
    changed_list = [change_list[-1]] + change_list[:-1]

    permutation_copy[1:8] = changed_list
    return permutation_copy


def switch(permutation):
    permutation_copy = copy(permutation)
    change_list = permutation_copy[1:7]
    changed_list = [
        change_list[-1],
        change_list[1],
        change_list[-3],
        change_list[2],
        change_list[-2],
        change_list[0]
        ]
    permutation_copy[1:7] = changed_list

    return permutation_copy


def generate_costs():
    costs = np.random.uniform(0,1,np.math.factorial(9)) * 115 + 10
    cost_rows = []
    cum_i = 0
    for i in range(9,-1, -1):
        row = [0] * (10 - i) + costs[cum_i:cum_i + i].tolist()
        cost_rows.append(row)
        cum_i += i

    cost_matrix_ = np.stack(cost_rows)
    cost_matrix = cost_matrix_.T + cost_matrix_

    return cost_matrix


def calculate_cost(permutation, cost_matrix):
    permutation_reverse = copy(permutation)
    permutation_reverse.reverse()

    total_costs = 0
    for _ in range(9):
        i = permutation_reverse.pop()
        j = permutation_reverse[-1]
        total_costs += cost_matrix[i,j]

    return total_costs


permutation = [0,1,2,3,4,5,6,7,8,9]
cost_matrix = generate_costs()
events = [0]*75 + [1]*15 + [2]*15
T = 70
c_b = 1
lmbd = 0.95
repls = 5
iters = 8000


for repl in range(repls):
    np.random.seed(repl)
    permutation = [0,1,2,3,4,5,6,7,8,9]
    current_cost = calculate_cost(permutation, cost_matrix)

    for iter in range(iters):
        decay_factor = (iter+1) // 40
        T_iter = T * lmbd**(decay_factor)

        operation = np.random.choice(events, 1).item()
        if operation == 0:
            new_permutation = inverse(permutation)
            
        elif operation == 1:
            new_permutation = translate(permutation)

        elif operation == 2:
            new_permutation = switch(permutation)
        
        new_cost = calculate_cost(new_permutation, cost_matrix)

        delta = new_cost - current_cost

        if delta < 0:
            current_cost = new_cost
            permutation = new_permutation

        else:
            p = np.exp(-delta / (c_b * T_iter))
            trial = bernoulli.rvs(size=1, p=p).item()

            if trial:
                current_cost = new_cost
                permutation = new_permutation

    print(f"[Replication {repl+1}] permutation: {permutation} / cost: {current_cost}")
