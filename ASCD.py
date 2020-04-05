from sympy import symbols, Matrix
import numpy as n
import json
import time

start = time.time()
with open('data.json', 'r') as read_file:
    data = json.load(read_file)


end = time.time()
print(end - start)
