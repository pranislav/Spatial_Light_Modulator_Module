import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from random import random


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
    while error > tolerance and n < max_loops:
        n+=1
        B = A/abs(A) # our source amplitude is 1 everywhere
        C = fftshift(fft2(B))
        D = np.abs(target) * C/abs(C)
        A = (ifft2(ifftshift(D)))
        exp_tar = np.abs(C) # np.abs(fftshift(fft2(A/abs(A))))
        scale = np.sqrt(255)/exp_tar.max()
        exp_tar *= scale
        error = error_f(exp_tar**2, target**2, space_norm)
        error_evolution.append(error)
        if n % 10 == 0: print("-", end='')
    print()
    printout(error, n, error_evolution, "asdf", plot_error)
    phase_for_slm = np.angle(A) * 255 / (2*np.pi) # converts phase to color value, input for SLM
    exp_tar = exp_tar**2
    exp_tar_for_slm = exp_tar * 255/np.amax(exp_tar) # what the outcome from SLM should look like
    return phase_for_slm, exp_tar_for_slm


def GS_pure(target: np.array, tolerance: float, max_loops: int, plot_error: bool=False) -> np.array:
    '''classical Gerchberg-Saxton algorithm
    produces input for SLM for creating 'target' image
    '''

    w, l = target.shape
    space_norm = w * l
    error = tolerance + 1
    error_evolution = []
    n = 0
    A = ifft2(target)
    while error > tolerance and n < max_loops:
        n+=1
        B = A/abs(A) # our source amplitude is 1 everywhere
        C = fft2(B)
        D = np.abs(target) * C/abs(C)
        A = ifft2(D)
        exp_tar = np.abs(C) # np.abs(fftshift(fft2(A/abs(A))))
        scale = np.sqrt(255)/exp_tar.max()
        exp_tar *= scale
        error = error_f(exp_tar**2, target**2, space_norm)
        error_evolution.append(error)
        if n % 10 == 0: print("-", end='')
    print()
    printout(error, n, error_evolution, "asdf", plot_error)
    phase_for_slm = np.angle(A) * 255 / (2*np.pi) # converts phase to color value, input for SLM
    exp_tar = exp_tar**2
    exp_tar_for_slm = exp_tar * 255/np.amax(exp_tar) # what the outcome from SLM should look like
    return phase_for_slm, exp_tar_for_slm


def GD(demanded_output: np.array, learning_rate: float, enhance_mask: np.array,\
       mask_relevance: float, tolerance: float, max_loops: int, unsettle: bool=False, plot_error: bool=False):
    w, l = demanded_output.shape
    space_norm = w * l
    initial_input = generate_initial_input(l, w)
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input))
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2 # toto prip. zapocitat do grad. zostupu
        mask = 1 + mask_relevance * enhance_mask
        dEdF = ifft2(mask * med_output * (output - demanded_output**2))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = error_f(output, demanded_output**2, space_norm)
        error_evolution.append(error)
        i += 1
        if unsettle and i % int(max_loops / 3) == 0:
            learning_rate *= 2
        if i % 10 == 0: print("-", end='')
    printout(error, i, error_evolution, f"learning_rate: {learning_rate}", plot_error)
    phase_for_slm = complex_to_real_phase(input/abs(input))
    exp_tar_for_slm = output
    return phase_for_slm, exp_tar_for_slm


def error_f(actual, correct, norm):
    return np.sum((actual - correct)**2) / norm


def printout(error, loop_num, error_evol, label, plot_error):
    print(f"error: {error}")
    print(f"number of loops: {loop_num}")
    if plot_error:
        plt.plot(error_evol, label=label)
        plt.legend()
        plt.show()


def generate_initial_input(w, h):
    random_matrix = [[(np.sqrt(random()) + 1j*np.sqrt(random())) for _ in range(w)] for _ in range(h)]
    return np.array(random_matrix)

def complex_to_real_phase(complex_phase):
    return (np.angle(complex_phase) + np.pi) / (2*np.pi) * 255

def dEdX_complex(dEdF, x):
    rE, iE = dEdF.real, dEdF.imag
    rx, ix = x.real, x.imag
    ax = abs(x)
    re_res = rE * (1/ax - rx**2 / ax**3) + iE * (- (rx*ix) / ax**3)
    im_res = rE * (- (rx*ix) / ax**3) + iE * (1/ax - ix**2 / ax**3)
    return re_res + 1j * im_res



def GD_uniphase(demanded_output_real: np.array, learning_rate: float, tolerance: float, max_loops, plot_error: bool=False):
    '''returns hologram which should result in picture with unifrom phase
    '''
    w, l = demanded_output_real.shape
    space_norm = w * l
    initial_input = generate_initial_input(l, w)
    error_evolution_re = []
    error_evolution_im = []
    norm = np.amax(demanded_output_real)
    demanded_output = two_d_array_to_complex(demanded_output_real)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input))
        normed_output = med_output / np.amax(med_output.real) * norm
        if i % 2 == 0:
            dEdF = ifft2(normed_output.real - demanded_output_real)
        else:
            dEdF = ifft2((normed_output.imag - demanded_output.imag) * 1j)
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error_re = error_f(normed_output.real, demanded_output_real, space_norm)
        error_im = error_f(normed_output.imag, demanded_output.imag, space_norm)
        error_evolution_re.append(error_re)
        error_evolution_im.append(error_im)
        i += 1
        if i % 10 == 0: print("-", end='')
    output = abs(med_output) **2 / np.amax(abs(med_output) **2) * norm**2 
    error = error_f(output, demanded_output_real**2, space_norm)
    printout(error, i, error_evolution_re, f"Re_part, learning_rate: {learning_rate}", plot_error)
    printout(error, i, error_evolution_im, f"Im_part, learning_rate: {learning_rate}", plot_error)
    phase_for_slm = complex_to_real_phase(input/abs(input))
    exp_tar_for_slm = output
    return phase_for_slm, exp_tar_for_slm


def two_d_array_to_complex(array: np.array):
    w, l = array.shape
    new_array = np.array([[0 + 0j for _ in range(l)] for _ in range(w)])
    for i in range(w):
        for j in range(l):
            new_array[i][j] = complex(array[i][j])
    return new_array


def GD_uniphaseII(demanded_output: np.array, learning_rate: float, tolerance: float, calib: float, angle: float, max_loops, plot_error: bool=False):
    k = calib
    w, l = demanded_output.shape
    space_norm = w * l
    initial_input = generate_initial_input(l, w)
    error_i_evolution = []
    error_ph_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input))
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2 # toto prip. zapocitat do grad. zostupu
        dE_intensity = 0 # med_output * (output - demanded_output**2)
        dE_phase = k * (np.angle(med_output) - angle) * 1j * med_output / np.abs(med_output)**2
        dEdF = ifft2(dE_intensity + dE_phase)
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error_i = error_f(output, demanded_output**2, space_norm)
        error_ph = k * error_f(np.angle(med_output), angle, space_norm)
        error = error_i + error_ph
        error_i_evolution.append(error_i)
        error_ph_evolution.append(error_ph)
        i += 1
        if i % 10 == 0: print("-", end='')
    printout(error, i, error_i_evolution, f"intensity error, learning_rate: {learning_rate}", plot_error)
    printout(error, i, error_ph_evolution, f"phase error, learning_rate: {learning_rate}", plot_error)
    phase_for_slm = complex_to_real_phase(input/abs(input))
    exp_tar_for_slm = output
    return phase_for_slm, exp_tar_for_slm