import json
import numpy as np
from sympy import symbols, Matrix

jon = {
    'n': 200,
    'p': 100,
    'k': 100
}

symbols0 = 'x0'

for i in range(1, jon['p']):
    symbols0 += ' x' + str(i)

jon['symbols'] = symbols0

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

# compute gradient
x = symbols(jon['symbols'])
f = (Matrix(jon['y']) - Matrix(jon['X']) * Matrix(x)).norm() ** 2

f_gradient = []
for i in range(jon['p']):
    f_gradient.append(f.diff(x[i]))

# make list of Add objects serializable
jon['f_gradient'] = [str(f_gradient[i]) for i in range(len(f_gradient))]

with open('data.json', 'w') as write_file:
    json.dump(jon, write_file)
