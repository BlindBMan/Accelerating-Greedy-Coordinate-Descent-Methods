import json
import numpy as np
from numba import jit


EPSILON = 0.000001

jon = {
    'n': 200,
    'p': 100,
    'k': 1000
}


X_bar = np.random.normal(0, 1, (jon['n'], jon['p']))

U, D_bar, V = np.linalg.svd(X_bar, full_matrices=False)
D_bar = np.diag(D_bar)
D = np.interp(D_bar, (D_bar.min(), D_bar.max()), (1 / np.sqrt(jon['k']), 1))

X = U @ D @ V
jon['X'] = X.tolist()

B_star = np.random.normal(0, 1, jon['p'])
y_mean = X @ B_star
y = np.random.normal(y_mean, 1)
jon['y'] = y.tolist()


@jit(nopython=True)
def theta_k(prev_theta):
    return (prev_theta ** 2 * (np.sqrt(prev_theta ** 2 + 4) - 1)) / 2


@jit(nopython=True)
def compute_theta():
    theta1 = [1]
    for k in range(1, 10000):
        thetak = theta_k(theta1[k - 1]) + EPSILON
        theta1.append(thetak)

    return theta1


theta = compute_theta()
jon['theta'] = theta[:jon['k']]

with open('data_1000.json', 'w') as write_file:
    json.dump(jon, write_file)