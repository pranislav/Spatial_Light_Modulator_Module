import numpy as np
from random import random
from scipy.fft import fft
import matplotlib.pyplot as plt
from slm.gluposti.der_E_wrt_input import dCdF_fun, vec_to_complex, complex_to_vec

dim = 101
input = np.array([np.exp(-20j * k) + np.exp(10j * k) for k in range(dim)]) #np.array([random() + 1j * random() for x in range(dim)])

def my_ft(x):
    d = len(x)
    y = [0 for _ in range(d)]
    for i in range(d):
        for j in range(d):
            y[i] += x[j] * np.exp(-1j * i *j * 2*np.pi / d)
    return np.flip(cycl_rot(y))

def cycl_rot(x):
    x = list(x)
    x.append(x[0])
    return np.array(x[1:])

def dft(x):
    y = np.zeros(np.array(x).shape)
    for j in range(len(x)):
        for k in range(len(x)):
            for r in range(2):
                for l in range(2):
                    arg = 2*np.pi/len(x) * j*k
                    dCdF = dCdF_fun(r, l, arg)
                    y[j][l] += x[k][r] * dCdF
    return vec_to_complex(y)


def dftc(x):
    y = np.zeros(np.array(x).shape)
    for j in range(len(x)):
        for k in range(len(x)):
            arg = 2*np.pi/len(x) * j*k
            dCdF = np.exp(-1j*arg) # 1j * np.sin(arg) + np.cos(arg) 
            y[j] += x[k] * dCdF
    return y


def dCdF_fun(r: int, l: int, arg: float) -> float:
    if r == 0:
        if l == 0:
            return np.cos(arg)
        else:
            return np.sin(arg)
    if r == 1:
        if l == 0:
            return - np.sin(arg)
        else:
            return np.cos(arg)


plt.plot(dft(complex_to_vec(input)), label="my")
plt.plot(dftc(input), label="fft")
plt.legend()
plt.show()
