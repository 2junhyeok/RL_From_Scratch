import numpy as np

x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

e = np.sum(x * pi)
print('E_pi[x]: ',e)

n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
mean = np.mean(samples)
var = np.var(samples)
print(f"MC: {mean} (var: {var})")


b = np.array([1/3, 1/3, 1/3])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))# [0, 1, 2]
    i = np.random.choice(idx, p = b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)
    
mean = np.mean(samples)
var = np.var(samples)
print(f'important sampling: {mean} (var: {var}')

b = np.array([0.2, 0.2, 0.6])
n = 100
samples = []

for _ in range(n):
    idx = np.arange(len(b))
    i = np.random.choice(idx, p=b)
    s = x[i]
    rho = pi[i] / b[i]
    samples.append(rho * s)

mean = np.mean(samples)
var = np.var(samples)
print(f'modified important sampling: {mean} (var: {var})')