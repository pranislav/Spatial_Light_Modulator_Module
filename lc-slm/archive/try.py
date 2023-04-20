import numpy as np
import PIL.Image as im
from PIL import Image, ImageOps

a = np.array([[i * j for i in range(1500)] for j in range(1000)])
r = np.amax(a)
k = a.max()

# print(a)
# print(r)
# print(k)

z = -1 - 0.1j
# print(np.angle(z))


