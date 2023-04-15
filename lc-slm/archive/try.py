import numpy as np

a = np.array([[i*j for i in range(6)] for j in range(4)])
r = np.amax(a)
k = a.max()

# print(a)
# print(r)
# print(k)

z = -1 - 0.1j
print(np.angle(z))