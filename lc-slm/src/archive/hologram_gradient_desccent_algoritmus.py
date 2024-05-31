from slm.gluposti.der_E_wrt_input import dEdX_real, complex_to_vec as ctv, dEdX_complex, dEdX_complex_II
import slm.gluposti.der_E_wrt_input as d
import numpy as np
from scipy.fft import fft, ifft
from random import random 
import matplotlib.pyplot as plt
import time

dim = 100
demanded_output = np.array([np.sin(x/20)**2 for x in range(dim)]) # d
input = np.array([random() + 1j * random() for _ in range(dim)]) #np.exp(20j*x)**2random() + 1j * random() # x


def GD_for_hologram(initial_input: np.array, demanded_output: np.array,\
        learning_rate: float, tolerance: float):
    error_evolution = []
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance:
        med_output = fft(input/abs(input))
        output = abs(med_output)/max(abs(med_output))
        output = output**2
        dEdF = ifft(med_output * (output - demanded_output))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        i += 1
        input -= learning_rate * dEdX
        error = sum((output - demanded_output)**2)/len(input)
        if i%100 == 0: print(error)
        error_evolution.append(error)
    return input/abs(input), output, error_evolution


def cycl_rot(x):
    x = list(x)
    x.append(x[0])
    return np.array(x[1:])




right_input, its_output, error_evolution = GD_for_hologram(input, demanded_output, 0.01, 1e-3)
plt.figure()
plt.plot(its_output, label="output")
plt.plot(demanded_output, label="demanded")
plt.legend()

plt.figure()
plt.plot(error_evolution)

plt.show()



if __name__ == "__main__":
    pass