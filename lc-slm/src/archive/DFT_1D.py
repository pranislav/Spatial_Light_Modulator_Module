import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from random import random


def norm_derivative_fun(x: np.array) -> np.array:
    Rx = np.real(x)
    Ix = np.imag(x)
    norm = abs(x)
    real_part = 1/norm - 1/(2 * norm**3) * (Rx**2 + Rx*Ix)
    imaginary_part = 1/norm - 1/(2 * norm**3) * (Ix**2 + Ix*Rx)
    return real_part + 1j * imaginary_part


def norm_derivative_fun2(x):
    return 1/abs(x) - x**2/(abs(x)**3)


source = np.array([complex(random()) for _ in range(100)])
target = np.array([complex(np.sin(i/10)) for i in range(100)])
tolerance = 100
learning_rate = 0.001
error = tolerance + 1
while error > tolerance:
    error_derivative_med = fft(source) - target
    norm_derivative = norm_derivative_fun2(source)
    change = fft(error_derivative_med) * norm_derivative
    source -= learning_rate * change
    error = sum(abs(fft(source/abs(source)) - target)**2)
    print(error)


plt.plot(np.real(fft(source)))
plt.show()
