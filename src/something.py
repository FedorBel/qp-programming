import numpy as np
import scipy as sp
from scipy import sparse


# a = np.array([[1, 0], [2, 0], [0, 3]])
# b = np.array([5, 6, 7])

# print(sparse.eye(1))

q1 = np.array([2, 0.5])
q1 = q1 / np.linalg.norm(q1)
q2 = np.array([-0.5, 2])
q2 = q2 / np.linalg.norm(q2)
print(q1.T @ q2)
