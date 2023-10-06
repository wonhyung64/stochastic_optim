#%%
import numpy as np


## Ex) 4.2
#%%
def init_theta():
    return np.ones([2])


def compute_grad(theta):
    t1, t2 = theta
    grad_t1 = 4 * np.power(t1, 3) + 2 * t1 + t2
    grad_t2 = t1 + 2 * t2
    return np.array(grad_t1, grad_t2)


def compute_loss(theta):
    t1, t2 = theta
    return np.power(t1, 4) + np.power(t1, 2) + t1 * t2 + np.power(t2, 2)


def generate_noise(sigma):
    return np.random.multivariate_normal(np.zeros([2]), np.identity(2) * sigma**2)


def generate_gain_seq(iter, power_factor):
    return 0.1 / np.power(iter + 1, power_factor)


def update_theta(theta, grad, gain_seq):
    return theta - gain_seq * grad


max_iter = 1000
max_repl = 200
power_factor_list = [1., .501]
sigma_list = [0.1, 1.0]

for sigma in sigma_list:

    for power_factor in power_factor_list:
        results = []

        for seed in range(max_repl):
            np.random.seed(seed)
            theta = init_theta()

            for iter in range(1, max_iter+1): 
                noise = generate_noise(sigma)
                grad = compute_grad(theta) + noise
                gain_seq = generate_gain_seq(iter, power_factor)
                theta = update_theta(theta, grad, gain_seq)

            terminal_loss = compute_loss(theta)
            results.append(terminal_loss)
        
        print(f"[Sigma: {sigma} / Gain seq: {power_factor}] mean terminal loss: {np.round(np.mean(results), 5)}")


## Ex) 4.5
#%%
def generate_x_pairs(max_sample_num=1000):
    c = np.random.randint(1, 11, size=max_sample_num)
    w = np.random.randint(11, 111, size=max_sample_num)
    return np.stack([w,c], 1)

def init_theta():
    return np.array([1., .5])


def generate_gain_seq(iter):
    return 0.0015 / np.power(iter + 100, .501) 


def generate_noise_1():
    return np.random.normal(0, 5**2)


def generate_noise_2(optimal_theta, x):
    h = compute_h(optimal_theta, x)
    w = np.random.normal(0, 1)
    return 0.2 * h * w


def compute_h(theta, x):
    lmbd, beta = theta
    w, c = x
    return lmbd * np.power(c, beta) * np.power(w, 1-beta)


def compute_euclidian(theta, optimal_theta):
    return np.sqrt(np.sum(np.power(theta - optimal_theta, 2)))


def compute_grad(theta, x):
    lmbd, beta = theta
    w, c = x
    grad_lmbd = w * np.power(c / w, beta)
    grad_beta = lmbd * w * np.power(c / w, beta) * np.log(c / w)
    
    return np.array([grad_lmbd, grad_beta])


def update_theta(theta, gain_seq, grad, x, z):
    h = compute_h(theta, x)
    return theta - gain_seq * grad * (h - z)


max_iter = 1000
max_repl = 5
optimal_theta = np.array([2.5, 0.7])

x = generate_x_pairs()
for noisy_type in ["Independent", "Dependent"]:

    for seed in range(max_repl):
        theta = init_theta()
        np.random.seed(seed)
        np.random.shuffle(x)
        
        for iter, x_sample in enumerate(x):
            gain_seq = generate_gain_seq(iter+1)
            
            if noisy_type == "Independent":
                noise = generate_noise_1()
            else:
                noise = generate_noise_2(optimal_theta, x_sample)

            z = compute_h(optimal_theta, x_sample) + noise
            grad = compute_grad(theta, x_sample)

            theta = update_theta(theta, gain_seq, grad, x_sample, z)
        print(f"[{noisy_type} Noise / Replication {seed+1}] theta = {theta}")


## Ex) 4.15
#%%
def init_theta():
    return np.ones([2])


def compute_grad(theta):
    t1, t2 = theta
    grad_t1 = 4 * np.power(t1, 3) + 2 * t1 + t2
    grad_t2 = t1 + 2 * t2
    return np.array(grad_t1, grad_t2)


def compute_loss(theta):
    t1, t2 = theta
    return np.power(t1, 4) + np.power(t1, 2) + t1 * t2 + np.power(t2, 2)


def generate_noise(sigma):
    return np.random.multivariate_normal(np.zeros([2]), np.identity(2) * sigma**2)


def generate_gain_seq(iter):
    return 0.1 / np.power(iter + 1, 0.501)


def update_theta(theta, grad, gain_seq):
    return theta - gain_seq * grad


max_iter = 1000
max_repl = 50
sigma = 1.0

results = []
for seed in range(max_repl):
    np.random.seed(seed)
    theta = init_theta()

    theta_list = [theta]
    for iter in range(1, max_iter+1): 
        noise = generate_noise(sigma)
        grad = compute_grad(theta) + noise
        gain_seq = generate_gain_seq(iter)
        theta = update_theta(theta, grad, gain_seq)
        theta_list.append(theta)

    standard_loss = compute_loss(theta)
    iter100_avg_loss = compute_loss(np.mean(theta_list[-100:], axis=0))
    iter_avg_loss = compute_loss(np.mean(theta_list, axis=0))
    results.append(np.array([standard_loss, iter100_avg_loss, iter_avg_loss]))


results_arr = np.stack(results, 0)
experiments = ["Standard", "Iterative Averaging recent 100", "Iterative Averaging"]
for i in range(3):
    print(f"[{experiments[i]}] Terminal loss: {np.round(np.mean(results_arr[:,i]), 4)} ± {np.round(np.std(results_arr[:,i], ddof=1), 4)}")

