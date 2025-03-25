import numpy as np

def sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x

if __name__ == "__main__":
    trial = 1000

    samples = []
    for _ in range(trial):# 한 번에 계산
        s = sample()
        samples.append(s)
    V = sum(samples) / len(samples)
    print(V)

    samples = []
    for _ in range(trial):# 추가마다 계산
        s = sample()
        samples.append(s)
        V = sum(samples) / len(samples)
        print(V)

    V, n = 0, 0
    for _ in range(trial):# 증분 계산
        s = sample()
        n +=1
        V = V + (s - V) / n
        print(V)