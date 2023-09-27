## Ex) 2.9
#%%
import numpy as np


def compute_loss(p, theta, B):
    loss = np.sum(np.power(theta[:p//2-1], 4)) + (theta @ B @ theta.T)

    return loss


def generate_B(p):
    off_diagonal = 0.5
    sigma = np.ones([p, p]) * off_diagonal
    np.fill_diagonal(sigma, 1., wrap=True)
    
    return sigma


def generate_d_k(p, rho):
    mu = np.zeros(p)
    cov = np.identity(p) * (rho**2)
    d_k = np.random.multivariate_normal(mu, cov, 1)

    return d_k


def init_theta(p):
    return np.ones(p)


max_iter = 100

### (a)
#%%
p = 20
rho_list = [.125, .25, .5, 1.]
max_repl = 40

B = generate_B(p)

for rho in rho_list:
    loss_list = []

    for seed in range(max_repl):
        np.random.seed(seed)
        theta = init_theta(p)
        current_loss = 1e+9

        for iter in range(max_iter):
            d_k = generate_d_k(p, rho)
            new_theta = theta + d_k
            loss = compute_loss(p, new_theta, B)
            if loss < current_loss:
                theta = new_theta
                current_loss = loss

        if type(current_loss) != np.float64:
            current_loss = current_loss[0,0]
        loss_list.append(current_loss)

    print(f"[rho {rho}] Mean Loss = {np.mean(loss_list)}")

### (b)
# %%
p = 2
rho_list = [.125, .25, .5, 1.]
max_repl = 40

B = generate_B(p)

for rho in rho_list:
    loss_list = []

    for seed in range(max_repl):
        np.random.seed(seed)
        theta = init_theta(p)
        current_loss = 1e+9

        for iter in range(max_iter):
            d_k = generate_d_k(p, rho)
            new_theta = theta + d_k
            loss = compute_loss(p, new_theta, B)
            if loss < current_loss:
                theta = new_theta
                current_loss = loss

        if type(current_loss) != np.float64:
            current_loss = current_loss[0,0]
        loss_list.append(current_loss)

    print(f"[rho {rho}] Mean Loss = {np.mean(loss_list)}")


## Ex) 2.13
# %%
import numpy as np


def compute_loss(theta, sigma, noisy=True):
    loss = np.power(theta[0], 4) + np.power(theta[0], 2) + (theta[0] * theta[1]) + np.power(theta[1], 2)
    if noisy:
        noise = generate_noise(sigma)
        obs_loss = loss + noise
    else:
        obs_loss = loss

    return obs_loss


def generate_noise(sigma):
    noise = np.random.normal(0, sigma**2, 1)[0]
    return noise


def generate_d_k():
    mu = np.zeros(2)
    cov = np.identity(2) * (0.125**2)
    d_k = np.random.multivariate_normal(mu, cov, 1)

    return d_k


def init_theta():
    return np.ones(2)

### (a)
#%%
sigma_list = [0.001, 0.01, 0.1, 1.0]
max_iter = 10000
max_repl = 40

avg_loss_num = 10

for method in ["avg", "theshold"]:

    for sigma in sigma_list:
        loss_list = []

        for seed in range(max_repl):
            np.random.seed(seed)
            theta = init_theta()
            current_loss = 1e+9

            for iter in range(max_iter):
                d_k = generate_d_k()
                new_theta = theta + d_k[0]

                if method == "avg":
                    obs_list = []
                    for _ in range(avg_loss_num):
                        obs_loss = compute_loss(new_theta, sigma)
                        obs_list.append(obs_loss)
                    loss = np.mean(obs_loss)
                
                else:
                    loss = (compute_loss(new_theta, sigma) + sigma*2)

                if loss < current_loss:
                    theta = new_theta
                    current_loss = loss

            if type(current_loss) != np.float64:
                current_loss = current_loss[0,0]
            loss_list.append(current_loss)

        print(f"[Method : {method} / sigma {sigma}] Mean Loss = {np.mean(loss_list)}")

### (b)
# %%
sigma_list = [0.001]
max_iter = 10000
max_repl = 40

avg_loss_num = 10

for method in ["avg", "theshold"]:

    for sigma in sigma_list:
        loss_list = []

        for seed in range(max_repl):
            np.random.seed(seed)
            theta = init_theta()
            current_loss = 1e+9

            for iter in range(max_iter):
                d_k = generate_d_k()
                new_theta = theta + d_k[0]
                loss = compute_loss(theta, sigma, noisy=False)

                if loss < current_loss:
                    theta = new_theta
                    current_loss = loss

            if type(current_loss) != np.float64:
                current_loss = current_loss[0,0]
            loss_list.append(current_loss)

        print(f"[Method : {method} / sigma {sigma}] Mean Loss = {np.mean(loss_list)}")
