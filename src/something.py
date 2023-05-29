import numpy as np
import scipy as sp
from scipy import sparse


a = np.array([[1, 0], [2, 0], [0, 3]])
b = np.array([5, 6, 7])
print(a)
print(b)
print(a + b.T)
