import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from random import random
import PIL.Image as im
from structs import gif_struct
from typing import List, Tuple
from cmath import phase


# function just for debugging old GS
def is_zero(a: np.array) -> List[Tuple[int, int]]:
    '''returns list of coordinates of zeros in a'''
    return [(i, j) for i in range(len(a)) for j in range(len(a[0])) if a[i][j] == 0]


def GS(target: np.array, path_to_incomming_intensity: str, tolerance: float, max_loops: int, gif_info: gif_struct, plot_error: bool=False) -> np.array:
    '''classical Gerchberg-Saxton algorithm
    produces input for SLM for creating 'target' image
    '''

    incomming_intensity = np.array(im.open(path_to_incomming_intensity, "L"))
    incomming_amplitude = np.sqrt(incomming_intensity)
    w, l = target.shape
    space_norm = w * l
    error = tolerance + 1
    error_evolution = []
    do_gif = gif_info.type
    skip_frames = gif_info.skip_frames
    n = 0
    A = ifft2(target)
    while error > tolerance and n < max_loops:
        B = incomming_amplitude * np.exp(1j * np.angle(A))
        C = fftshift(fft2(B))
        D = np.abs(target) * np.exp(1j * np.angle(C))
        A = (ifft2(ifftshift(D)))
        exp_tar = np.abs(C) # np.abs(fftshift(fft2(A/abs(A))))
        scale = np.sqrt(255)/exp_tar.max()
        exp_tar *= scale
        error = error_f(exp_tar**2, target**2, space_norm)
        error_evolution.append(error)
        if do_gif and n % skip_frames == 0:
            if do_gif == 'h':
                img = im.fromarray((np.angle(A) + np.pi) * 255 / (2*np.pi))
            if do_gif == 'i':
                exp_tar **= 2
                phase_for_slm = exp_tar * 255/np.amax(exp_tar)
                img = im.fromarray(phase_for_slm)
            img.convert("RGB").save(f"{gif_info.source_address}/{n // skip_frames}.jpg")
        n+=1
        if n % 10 == 0: print("-", end='')
    print()
    printout(error, n, error_evolution, "asdf", plot_error)
    phase_for_slm = (np.angle(A) + np.pi) * 255 / (2*np.pi) # converts phase to color value, input for SLM
    exp_tar = exp_tar**2
    exp_tar_for_slm = exp_tar * 255/np.amax(exp_tar) # what the outcome from SLM should look like
    return phase_for_slm, exp_tar_for_slm, n


def GS_for_moving_traps(target: np.array, tolerance: float, max_loops: int) -> np.array:
    '''GS adapted for creeating holograms for moving traps
    '''

    w, l = target.shape
    space_norm = w * l
    error = tolerance + 1
    error_evolution = []
    n = 0
    A = ifft2(target)
    # print(is_zero(A))
    while error > tolerance and n < max_loops:
        B = np.exp(1j * np.angle(A)) # our source amplitude is 1 everywhere
        C = fftshift(fft2(B))
        D = np.abs(target) * np.exp(1j * np.angle(C))
        A = (ifft2(ifftshift(D)))
        exp_tar = np.abs(C) # np.abs(fftshift(fft2(A/abs(A))))
        scale = np.sqrt(255)/exp_tar.max()
        exp_tar *= scale
        error = error_f(exp_tar**2, target**2, space_norm)
        error_evolution.append(error)
        n+=1
        if n % 10 == 0: print("-", end='')
    print()
    phase_for_slm = (np.angle(A) + np.pi) * 255 / (2*np.pi) # converts phase to color value, input for SLM
    exp_tar = exp_tar**2
    exp_tar_for_slm = exp_tar * 255/np.amax(exp_tar) # what the outcome from SLM should look like
    return phase_for_slm, exp_tar_for_slm, error_evolution

def GS_without_fftshift(target: np.array, tolerance: float, max_loops: int, plot_error: bool=False) -> np.array:
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


def GD(demanded_output: np.array, path_to_inicomming_intensity: str, learning_rate: float, enhance_mask: np.array,\
       mask_relevance: float, tolerance: float, max_loops: int, unsettle, gif_info: gif_struct, plot_error: bool=False):
    
    incomming_intensity = np.array(im.open(path_to_inicomming_intensity, "L"))
    incomming_amplitude = np.sqrt(incomming_amplitude)
    w, l = demanded_output.shape
    space_norm = w * l
    initial_input = generate_initial_input(l, w)
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    skip_frames = gif_info.skip_frames
    do_gif = gif_info.type
    i = 0
    print("computing hologram (one bar for 10 loops)", ' ')
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input) * incomming_amplitude)
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2 # toto prip. zapocitat do grad. zostupu
        mask = 1 + mask_relevance * enhance_mask
        dEdF = ifft2(mask * med_output * (output - demanded_output**2)) * incomming_amplitude
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = error_f(output, demanded_output**2, space_norm)
        error_evolution.append(error)
        if do_gif and i % skip_frames == 0:
            if do_gif == 'h':
                img = im.fromarray(complex_to_real_phase(input/abs(input)))
            if do_gif == 'i':
                img = im.fromarray(output)
            img.convert("RGB").save(f"{gif_info.source_address}/{i // skip_frames}.jpg")
        i += 1
        if unsettle and i % int(max_loops / unsettle) == 0:
            learning_rate *= 2
        if i % 10 == 0: print("-", end='')
    printout(error, i, error_evolution, f"learning_rate: {learning_rate}", plot_error)
    phase_for_slm = complex_to_real_phase(input)
    exp_tar_for_slm = output
    return phase_for_slm, exp_tar_for_slm, i


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

def GD_for_moving_traps(demanded_output: np.array, initial_input: np.array, learning_rate: float=0.005,
       mask_relevance: float=10, tolerance: float=0.001, max_loops: int=50, unsettle=0):
    w, l = demanded_output.shape
    space_norm = w * l
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input))
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2 # toto prip. zapocitat do grad. zostupu
        mask = 1 + mask_relevance * demanded_output/255
        dEdF = ifft2(mask * med_output * (output - demanded_output**2))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = error_f(output, demanded_output**2, space_norm)
        error_evolution.append(error)
        i += 1
        if unsettle and i % int(max_loops / unsettle) == 0:
            learning_rate *= 2
        if i % 10 == 0: print("-", end='')
    print()
    phase_for_slm = complex_to_real_phase(input)
    exp_tar_for_slm = output
    init_input_for_next = input
    return phase_for_slm, init_input_for_next, exp_tar_for_slm, error_evolution