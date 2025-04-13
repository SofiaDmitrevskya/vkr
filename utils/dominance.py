import numpy as np

def find_dominated_pairs(X, epsilon=1e-2):
    pairs = []
    N = X.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.all(X[i] < X[j]):
                diff_norm = np.linalg.norm(X[j] - X[i])
                if diff_norm > epsilon:
                    pairs.append((i, j))
    return pairs
