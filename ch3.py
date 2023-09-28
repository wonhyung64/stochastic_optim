## 3.2
#%%
import numpy as np
import matplotlib.pyplot as plt


def generate_w_k():
    return np.random.normal(0,1,1)[0]


def generate_x_k(x):
    return -0.99 * x + generate_w_k()

def compute_grad(y, x, theta):
    return x*(x * theta - y)

def update_theta(theta, grad, a):
    return theta - a * grad

max_iter = 500
a = 0.005
x = 0
theta = 0

theta_list = [theta]
for _ in range(max_iter):
    x_new = generate_x_k(x)
    grad = compute_grad(x_new, x, theta)
    theta = update_theta(theta, grad, a)
    x = x_new
    theta_list.append(theta)


### (a)
#%%
plt.style.use('classic')
plt.plot(theta_list)
plt.xlabel("Iterations")
plt.ylabel("Estimates of beta")
plt.show()


### (b)
#%%
max_repl = 50
max_iter = 500
a = 0.005

theta_list = []
for seed in range(max_repl):
    np.random.seed(seed)
    x = 0
    theta = 0
    for _ in range(max_iter):
        x_new = generate_x_k(x)
        grad = compute_grad(x_new, x, theta)
        theta = update_theta(theta, grad, a)
        x = x_new
    theta_list.append(theta)

print(f"Estimates of Beta: {np.round(np.mean(theta_list),3)} ± {np.round(np.std(theta_list, ddof=1),3)} (n = 50)")

## 3.4
#%%
import numpy as np
import matplotlib.pyplot as plt


def generate_x_k(x):
    u = generate_u_k()
    return np.array([0.9, 0.5]) @ np.array(x, u).T + generate_w_k()


def generate_u_k():
    pass


def generate_d_k(iter):
    return np.sin(np.pi * iter+1 / 10)


def generate_w_k():
    return np.random.normal(0, 0.25**2, 1)[0]


def compute_grad(Y, x, iter, theta):
    u = generate_u_k()
    return np.array([x,u]) @ ((np.array([x, u]) @ theta) - Y)


def update_theta(theta, grad, a):
    return theta - a * grad




max_iter = 200
a = 0.001
x = np.zeros(1)
theta = np.ones(2)

theta_list = [theta]
for iter in range(max_iter):
    x_new = generate_x_k(x)
    grad = compute_grad(x_new, x, theta)
    theta = update_theta(theta, grad, a)
    x = x_new
    theta_list.append(theta)



## 3.8
#%%
import numpy as np


def generate_h_k(iter):
    return np.array([iter, 1])


def generate_v_k():
    return np.random.normal(0, 1, 1)[0]


def generate_z_k(h_k):
    v_k = generate_v_k()
    return np.inner(h_k, np.array([1, 2])) + v_k


def compute_p_k(p_k_inv, h_k):
    return np.linalg.inv(p_k_inv + np.outer(h_k, h_k))


def update_theta(theta, p_k, p_k_inv, h_k, z_k):
    return np.inner(p_k, (np.inner(p_k_inv, theta) + (h_k * z_k)))


n_list = [10, 100, 1000, 10000]
p_k = np.zeros([2, 2])
np.fill_diagonal(p_k, 1., wrap=True)
p_k *= 100
theta = np.zeros([2])

for iter in range(1, max(n_list)+1):
    h_k = generate_h_k(iter)
    z_k = generate_z_k(h_k)
    p_k_inv = np.linalg.inv(p_k)
    p_k = compute_p_k(p_k_inv, h_k)
    theta = update_theta(theta, p_k, p_k_inv, h_k, z_k)
    if iter in n_list:
        print(f"[Iteration {iter}] theta = {theta}")
    


## 3.10
# %%
import numpy as np


def generate_h_k(iter):
    return np.array([iter, 1])


def generate_v_k(c):
    return np.random.normal(0, c, 1)[0]


def generate_z_k(h_k, c):
    if h_k[0] % 2:
        v_k = generate_v_k(1)
    else: 
        v_k = generate_v_k(c)
    return np.inner(h_k, np.array([1, 2])) + v_k


#%%
def compute_p_k(p_k, h_k):
    numer = np.outer(np.inner(p_k, h_k), h_k) @ p_k
    denom = 1 + np.inner(np.inner(h_k, p_k), h_k)
    return p_k - numer / denom


def update_theta(theta, p_k, h_k, z_k):
    return theta - (p_k @ h_k) * (np.inner(h_k, theta) - z_k)


n_list = [10, 100, 1000, 10000, 100000]
c_list = [100, 1000]

theta = np.zeros([2])
p_k = np.zeros([2, 2])
np.fill_diagonal(p_k, 1., wrap=True)
p_k *= 100

print("===========UNWEIGHTED RLS===========")
for c in c_list:
    for iter in range(1, max(n_list)+1):
        h_k = generate_h_k(iter)
        z_k = generate_z_k(h_k, c)
        p_k = compute_p_k(p_k, h_k)
        theta = update_theta(theta, p_k, h_k, z_k)
        if iter in n_list:
            print(f"[C = {c} / Iteration {iter}] theta = {theta}")
print()

#%%
def compute_p_k(p_k, h_k, c):
    numer = np.outer(np.inner(p_k, h_k), h_k) @ p_k
    if h_k[0] % 2 :
        weight = 1
    else:
        weight = c
    denom = weight + np.inner(np.inner(h_k, p_k), h_k)
    return p_k - numer / denom


def update_theta(theta, p_k, h_k, z_k, c):
    if h_k[0] % 2 :
        weight = 1
    else:
        weight = c
    return theta - (p_k @ h_k) / weight * (np.inner(h_k, theta) - z_k)


n_list = [10, 100, 1000, 10000, 100000]
c_list = [100, 1000]

theta = np.zeros([2])
p_k = np.zeros([2, 2])
np.fill_diagonal(p_k, 1., wrap=True)
p_k *= 100

print("==========WEIGHTED RLS==========")
for c in c_list:
    for iter in range(1, max(n_list)+1):
        h_k = generate_h_k(iter)
        z_k = generate_z_k(h_k, c)
        p_k = compute_p_k(p_k, h_k, c)
        theta = update_theta(theta, p_k, h_k, z_k, c)
        if iter in n_list:
            print(f"[C = {c} / Iteration {iter}] theta = {theta}")
print()
# %%

