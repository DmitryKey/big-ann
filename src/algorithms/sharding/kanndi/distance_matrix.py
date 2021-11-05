import numpy as np
from scipy.spatial import distance_matrix

a = np.zeros((3, 2))
b = np.ones((4, 2))

print("a")
print(a)
print("b")
print(b)

dist = distance_matrix(a, b)

#print(dist)

a1D = np.array([[2.3509887e-38, 9.6263779e-38, 3.7615842e-37]])
print(a1D)
b1D = np.array([[9.8565270e-35, 2.4041676e-27, 2.8025969e-45]])
print(b1D)

dist = distance_matrix(a1D, b1D)
print(dist)