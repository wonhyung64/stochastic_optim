#%%
import numpy as np
from scipy.stats import bernoulli

#%%
## 7.3

def generate_lr(iter):
    return 0.5 / np.power(iter + 1 + 50, 0.602)

def generate_ck(iter):
    return 0.1 / np.power(iter + 1, 0.101)


def calculate_loss(theta, B, ck, nabla):
    measure = theta @ B @ B.T @ theta.T + 0.1 * np.sum((B.T @ theta.T) @ (B.T @ theta.T) * (B.T @ theta.T)) + 0.01 * np.sum((B.T @ theta.T) @ (B.T @ theta.T) * (B.T @ theta.T) @ (B.T @ theta.T))
    return measure + ck * nabla


def generate_nabla1():
    rv = bernoulli.rvs(size=1, p=0.5).item()
    if rv:
        return 1
    else:
        return -1


def generate_nabla1():
    rv = bernoulli.rvs(size=10, p=0.5)
    return rv + rv - 1


def generate_nabla2():
    return np.random.uniform(-np.sqrt(3), np.sqrt(3), 10)


def generate_nabla3():
    return np.random.normal(0, 1, 10)


def generate_perturb():
    return np.random.normal(0, 0.01, 1).item()


p = 10
iters = 2000
B = np.triu(np.ones([p,p])) / p

theta = np.ones(p)
for iter in range(iters):
    lr = generate_lr(iter)
    ck = generate_ck(iter)
    nabla = generate_nabla1()
    grad = ((calculate_loss(theta, B, ck, nabla)+generate_perturb()) - calculate_loss(theta, B, ck, -nabla)+generate_perturb()) / (2*ck) @ (1 / nabla)

    new_theta = theta - lr * grad

print(f"[Perturbation 1] theta: \n{new_theta}")


theta = np.ones(p)
for iter in range(iters):
    lr = generate_lr(iter)
    ck = generate_ck(iter)
    nabla = generate_nabla2()
    grad = ((calculate_loss(theta, B, ck, nabla)+generate_perturb()) - calculate_loss(theta, B, ck, -nabla)+generate_perturb()) / (2*ck) @ (1 / nabla)

    new_theta = theta - lr * grad

print(f"[Perturbation 2] theta: \n{new_theta}")


theta = np.ones(p)
for iter in range(iters):
    lr = generate_lr(iter)
    ck = generate_ck(iter)
    nabla = generate_nabla3()
    grad = ((calculate_loss(theta, B, ck, nabla)+generate_perturb()) - calculate_loss(theta, B, ck, -nabla)+generate_perturb()) / (2*ck) @ (1 / nabla)

    new_theta = theta - lr * grad

print(f"[Perturbation 3] theta: \n{new_theta}")
