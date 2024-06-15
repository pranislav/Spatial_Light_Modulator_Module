'''algorithms for generating binary holograms
usecase DMD (digital micromirror device)
'''


from scipy.fft import fft2, ifft2
import numpy as np
import dmd_constants as c
from incident_angle import the_angle
import random
import copy


def binary_GD(demanded_output: np.array, phase_profile: np.array,
                        tolerance: float=0.001, max_loops: int=50):
    w, l = demanded_output.shape
    binary_mask = np.random.randint(0, 2, (w, l))
    space_norm = w * l
    error_evolution = []
    norm = np.amax(demanded_output)
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(phase_profile * binary_mask)
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2
        dEdF = ifft2(med_output * (output - demanded_output**2))
        min_i, min_j = np.unravel_index(np.argmin(abs(dEdF)), dEdF.shape)
        # print(min_i, min_j, end=' ')
        difference_F = phase_profile * (-1)**binary_mask
        difference_E = dEdF * difference_F
        # binary_mask = update_binary_mask_single(binary_mask, difference_E[i, i], i, i)
        binary_mask = update_binary_mask_single(binary_mask, True, min_i, min_j)
        error = error_f(output, demanded_output**2, space_norm)
        error_evolution.append(error)
        i += 1
        if i % 10 == 0: print("-", end='')
    print()
    hologram = binary_mask
    exp_tar_for_dmd = output
    return hologram, exp_tar_for_dmd, error_evolution


def naive_alg(demanded_output: np.array, phase_profile: np.array, tolerance: float=0.001, max_loops: int=50):
    w, l = demanded_output.shape
    binary_mask = np.random.randint(0, 2, (w, l))
    space_norm = w * l
    error_evolution = []
    norm = np.amax(demanded_output)
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        k, m = random_tuple(w, l)
        output = F(binary_mask, phase_profile, norm)
        E1 = error_f(output, demanded_output**2, space_norm)
        output = F(update_binary_mask_single(binary_mask, k, m), phase_profile, norm)
        E2 = error_f(output, demanded_output**2, space_norm)
        if E2 < E1:
            binary_mask = update_binary_mask_single(binary_mask, k, m)
            error_evolution.append(E2)
        else:
            error_evolution.append(E1)
        i += 1
        if i % 10 == 0: print("-", end='')
    print()
    hologram = binary_mask
    exp_tar_for_dmd = output
    return hologram, exp_tar_for_dmd, error_evolution

def random_tuple(w, l):
    return random.randint(0, w-1), random.randint(0, l-1)

def F(binary_mask, phase_profile, norm):
    med_output = fft2(phase_profile * binary_mask)
    output_unnormed = abs(med_output) **2
    return output_unnormed / np.amax(output_unnormed) * norm**2



def error_f(actual, correct, norm):
    return np.sum((actual - correct)**2) / norm


def update_binary_mask_single(binary_mask, cond, i, j):
    if not cond:
        return binary_mask
    binary_mask = copy.deepcopy(binary_mask)
    binary_mask[i][j] = not binary_mask[i][j]
    return binary_mask

def update_binary_mask_multi(binary_mask, difference_E):
    for i in range(len(binary_mask)):
        for j in range(len(binary_mask[0])):
            if random.randint(0, 10000) == 0 and difference_E[i][j] < 0:
                binary_mask[i][j] = not binary_mask[i][j]
    return binary_mask


def phase_profile_straight(w, h):
    phase = c.diagonal_spacing * 2*np.pi / c.wavelength * np.sin(the_angle)
    return np.array([[np.exp(1j * k * phase) for k in range(w)] for _ in range(h)])

def phase_profile_diagonal(w, h):
    phase = c.diagonal_spacing * 2*np.pi / c.wavelength * np.sin(the_angle)
    return np.array([[np.exp(1j * (k + l) / 2 * phase) for k in range(w)] for l in range(h)])