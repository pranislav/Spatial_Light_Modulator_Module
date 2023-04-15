import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


def GS(target: np.array, tolerance: float, max_loops: int, plot_error: bool=False) -> np.array:
    '''classical Gerchberg-Saxton algorithm
    produces input for SLM for creating 'target' image
    '''

    w, l = target.shape
    space_norm = w * l
    error = tolerance + 1
    error_evolution = []
    n = 0
    A = ifft2(target)
    while error > tolerance or n > max_loops:
        n+=1
        B = A/abs(A) # our source amplitude is 1 everywhere
        C = fftshift(fft2(B))
        D = np.abs(target) * C/abs(C)
        A = (ifft2(ifftshift(D)))
        exp_tar = np.abs(fftshift(fft2(A/abs(A))))
        scale = np.sqrt(255)/max(exp_tar.max(0))
        exp_tar *= scale
        error = np.sum((target - exp_tar)**2) / space_norm
        error_evolution.append(error)
    if plot_error:
        error_plot(error_evolution, f"asdf")
    print(f"error in GS: {error}")
    print(f"number of loops in GS: {n}")
    phase_for_slm = np.angle(A) * 255 / (2*np.pi) # converts phase to color value, input for SLM
    exp_tar = exp_tar**2
    exp_tar_for_slm = exp_tar * 255/np.amax(exp_tar) # what the outcome from SLM should look like
    return phase_for_slm, exp_tar_for_slm


def error_plot(error_evolution: list, label):
    plt.plot(error_evolution, label=label)
    plt.legend()
    plt.show()

