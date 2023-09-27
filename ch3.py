## 3.2
#%%
import numpy as np


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


for _ in range(max_iter):
    x_new = generate_x_k(x)
    grad = compute_grad(x_new, x, theta)
    theta = update_theta(theta, grad, a)
    x = x_new






## 3.4
## 3.8
## 3.10