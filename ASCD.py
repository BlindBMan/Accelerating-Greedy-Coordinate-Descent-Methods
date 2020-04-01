from sympy import *
import numpy as np
import ast
import json

with open('data.json', 'r') as read_file:
    data = json.load(read_file)

x = symbols(data['symbols'])
f = (Matrix(data['y']) - Matrix(data['X']) * Matrix(x)).norm() ** 2

