import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x**2)

def monte_carlo(N):
    samples = np.random.uniform(0, 2, N)
    integral = (2 / N) * np.sum(f(samples))
    return integral

def importance_sampling(N):
    def pdf(x):
        return np.exp(-x**2) / (np.sqrt(np.pi) * (0.5))  # Gaussian PDF scaled to match range
    
    def sample_gaussian(N):
        return np.random.normal(0, 0.5, N)  # Samples from Gaussian centered at 0 with std 0.5
    
    samples = sample_gaussian(N)
    samples = samples[(samples >= 0) & (samples <= 2)]  # Restrict to [0,2]
    weights = f(samples) / pdf(samples)
    integral = np.mean(weights)
    return integral

true_value = 0.882081
N_values = np.logspace(1, 4, 20, dtype=int)
mc_errors = []
is_errors = []

for N in N_values:
    mc_errors.append(abs((monte_carlo(N) - true_value) / true_value))
    is_errors.append(abs((importance_sampling(N) - true_value) / true_value))

plt.figure(figsize=(8, 6))
plt.loglog(N_values, mc_errors, label='Monte Carlo Relative Error', marker='o')
plt.loglog(N_values, is_errors, label='Importance Sampling Relative Error (Gaussian)', marker='s')
plt.xlabel('Number of Samples')
plt.ylabel('Relative Error')
plt.legend()
plt.title('Monte Carlo vs Importance Sampling Relative Errors with Gaussian Weight')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

N = 10000
mc_result = monte_carlo(N)
is_result = importance_sampling(N)

print(f"Monte Carlo Estimation: {mc_result}")
print(f"Importance Sampling Estimation (Gaussian): {is_result}")
