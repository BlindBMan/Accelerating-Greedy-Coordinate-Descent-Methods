from sympy import symbols, Matrix
import numpy as np
import json
import time

start = time.time()
with open('data.json', 'r') as read_file:
    data = json.load(read_file)

theta_curr, theta_prev = symbols('t_curr t_prev')
theta_fun = (1 - theta_curr) * theta_prev ** 2 - theta_curr ** 2


end = time.time()
print(end - start)
