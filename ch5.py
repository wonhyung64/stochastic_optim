#%%
import numpy as np

## 5.8
# %%
def generate_x_pairs(max_sample_num=1000):
    c = np.random.randint(1, 11, size=max_sample_num)
    w = np.random.randint(11, 111, size=max_sample_num)
    return np.stack([w,c], 1)

def init_theta():
    return np.array([1., .5])


def generate_gain_seq(iter, start_gain):
    return start_gain / np.power(iter + 100, .501) 


def generate_noise_1():
    return np.random.normal(0, 5**2)


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

for seed in range(max_repl):
    theta = init_theta()
    np.random.seed(seed)
    np.random.shuffle(x)
    
    for iter, x_sample in enumerate(x):
        gain_seq = generate_gain_seq(iter+1, start_gain=0.00125)
        noise = generate_noise_1()

        z = compute_h(optimal_theta, x_sample) + noise
        grad = compute_grad(theta, x_sample)

        theta = update_theta(theta, gain_seq, grad, x_sample, z)
    print(f"[Case (i) / Replication {seed+1}] theta = {theta}")


for seed in range(max_repl):
    theta = init_theta()
    np.random.seed(seed)
    np.random.shuffle(x)
    x_sum = np.array([0,0])
    z_sum = 0.
    for iter, x_sample in enumerate(x):
        gain_seq = generate_gain_seq((iter+1) // 2, start_gain=0.0075)
        noise = generate_noise_1()

        z = compute_h(optimal_theta, x_sample) + noise
        if (iter + 1) % 2 != 0:
            x_sum += (x_sample)
            z_sum += z
            continue
        
        x_mean = x_sum / 2
        z_mean = z_sum / 2
        grad = compute_grad(theta, x_mean)

        theta = update_theta(theta, gain_seq, grad, x_mean, z_mean)

        x_sum = np.array([0,0])
        z_sum = 0.
    print(f"[Case (ii) / Replication {seed+1}] theta = {theta}")



for seed in range(max_repl):
    theta = init_theta()
    np.random.seed(seed)
    np.random.shuffle(x)
    for epoch in range(10):
        for iter, x_sample in enumerate(x):
            gain_seq = generate_gain_seq(epoch*1000 + iter+1, start_gain=0.00015)
            noise = generate_noise_1()

            z = compute_h(optimal_theta, x_sample) + noise
            grad = compute_grad(theta, x_sample)

            theta = update_theta(theta, gain_seq, grad, x_sample, z)
    print(f"[Case (iii) / Replication {seed+1}] theta = {theta}")