import json
import numpy as np
import matplotlib.pyplot as plt


EPSILON = 0.000001

with open('data_10000.json', 'r') as f:
    data = json.load(f)


# init L with diag of hessian matrix
def gen_L(X):
    L = []
    for k in range(data['p']):
        sum = 0
        for i in range(data['n']):
            sum += X[i][k] ** 2
        sum = sum * 2
        L.append(sum + EPSILON)

    return L


L = gen_L(data['X'])


def f(y, X, beta):
    val = 0
    for i in range(len(y)):
        x_mul_beta = 0
        for j in range(len(beta)):
            x_mul_beta += X[i][j] * beta[j]
        val += y[i] - x_mul_beta

    return val


# compute gradient
def gradient_k(y, X, beta, k):
    grad = 0
    for i in range(data['n']):
        x_plus_beta = 0
        for j in range(data['p']):
            x_plus_beta += X[i][j] * beta[j]
        grad += 2 * (y[i] - x_plus_beta) * (-X[i][k])

    return grad


def gradient(y, X, beta):
    f_gradient = []

    for kth in range(data['p']):
        f_gradient.append(np.abs(gradient_k(y, X, beta, kth)) / np.sqrt(L[kth]))

    return f_gradient.index(max(f_gradient))


# main algorithm
def alg(alg_type):
    xk = np.zeros(data['p'])
    zk = xk.copy()

    # EPSILON = 0.000001
    # L = 1 / (np.random.rand(data['p']) + EPSILON)

    func_values_xk = []
    func_values_zk = []

    for k in range(data['k']):
        xk1 = [(1 - data['theta'][k]) * x for x in xk]
        zk1 = [data['theta'][k] * z for z in zk]
        yk = [x + z for x, z in zip(xk1, zk1)]

        if alg_type == 'arcd':
            j1 = int(np.random.uniform(0, data['p']))
        elif alg_type == 'agcd' or alg_type == 'ascd':
            j1 = gradient(data['y'], data['X'], yk)

        ej1 = np.zeros(data['p'])
        ej1[j1] = gradient_k(data['y'], data['X'], yk, j1) / L[j1]
        xk = yk - ej1
        func_values_xk.append(f(data['y'], data['X'], xk))
        print(func_values_xk[-1])

        if alg_type == 'arcd' or alg_type == 'agcd':
            j2 = j1
        elif alg_type == 'ascd':
            j2 = int(np.random.uniform(0, data['p']))

        ej2 = np.zeros(data['p'])
        ej2[j2] = gradient_k(data['y'], data['X'], yk, j2) / (data['n'] * L[j2] * data['theta'][k])
        zk = zk - ej2
        func_values_zk.append(f(data['y'], data['X'], zk))
        print(func_values_zk[-1])

    return func_values_xk, func_values_zk


func_values_xk, func_values_zk = alg('arcd')

# plots

iters = np.array(range(data['k']))
plt.plot(iters, func_values_xk)
plt.show()
plt.plot(iters, func_values_zk)
plt.show()
