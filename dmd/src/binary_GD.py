from scipy.fft import fft2, ifft2
import numpy as np
import dmd_constants as c
from incident_angle import the_angle


def discrete_GD(demanded_output: np.array, phase_profile: np.array,
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
        difference_F = phase_profile * (-1)**binary_mask
        difference_E = dEdF * difference_F
        binary_mask = update_binary_mask(binary_mask, difference_E)
        error = error_f(output, demanded_output**2, space_norm)
        error_evolution.append(error)
        i += 1
        if i % 10 == 0: print("-", end='')
    print()
    hologram = binary_mask
    exp_tar_for_dmd = output
    return hologram, exp_tar_for_dmd, error_evolution


def error_f(actual, correct, norm):
    return np.sum((actual - correct)**2) / norm


def update_binary_mask(binary_mask, difference_E):
    count = 0
    for i in range(len(binary_mask)):
        for j in range(len(binary_mask[0])):
            if difference_E[i][j] < 0:
                binary_mask[i][j] = not binary_mask[i][j]
                count += 1
    # print(count, end=' ')
    return binary_mask


def phase_profile_straight(w, h):
    phase = c.diagonal_spacing * 2*np.pi / c.wavelength * np.sin(the_angle)
    return np.array([[np.exp(1j * k * phase) for k in range(w)] for _ in range(h)])

def phase_profile_diagonal(w, h):
    phase = c.diagonal_spacing * 2*np.pi / c.wavelength * np.sin(the_angle)
    return np.array([[np.exp(1j * (k + l) / 2 * phase) for k in range(w)] for l in range(h)])