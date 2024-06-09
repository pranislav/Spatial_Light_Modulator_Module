import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from random import random
import PIL.Image as im
from typing import List, Tuple
from cmath import phase


# function just for debugging old GS
def is_zero(a: np.array) -> List[Tuple[int, int]]:
    '''returns list of coordinates of zeros in a'''
    return [(i, j) for i in range(len(a)) for j in range(len(a[0])) if a[i][j] == 0]


def GS(target, args) -> np.array:
    '''classical Gerchberg-Saxton algorithm
    produces input for SLM for creating 'target' image
    '''

    incomming_intensity = np.array(im.open(args.path_to_incomming_intensity))
    incomming_amplitude = np.sqrt(incomming_intensity)
    w, l = target.shape
    target_amplitude = np.sqrt(target)
    space_norm = w * l
    norm = np.amax(target)
    error = args.tolerance + 1
    error_evolution = []
    n = 0
    A = ifft2(target_amplitude)
    while error > args.tolerance and n < args.max_loops:
        B = incomming_amplitude * np.exp(1j * np.angle(A))
        C = fftshift(fft2(B))
        D = np.abs(target_amplitude) * np.exp(1j * np.angle(C))
        A = (ifft2(ifftshift(D)))
        expected_outcome = np.abs(C)**2
        expected_outcome *= norm / expected_outcome.max()
        error = error_f(expected_outcome, target, space_norm)
        error_evolution.append(error)
        if args.gif and n % args.gif_skip == 0:
            if args.gif_type == 'h':
                img = im.fromarray((np.angle(A) + np.pi) * args.correspond_to2pi / (2*np.pi))
            if args.gif_type == 'i':
                img = im.fromarray(expected_outcome)

            img.convert("RGB").save(f"{args.gif_source_address}/{n // args.gif_skip}.png")
        n+=1
        if n % 10 == 0: print("-", end='')
    print()
    printout(error, n, error_evolution, "asdf", args.plot_error)
    phase_for_slm = (np.angle(A) + np.pi) * args.correspond_to2pi / (2*np.pi) # converts phase to color value, input for SLM
    return phase_for_slm, expected_outcome


def GS_for_moving_traps(target: np.array, tolerance: float, max_loops: int, correspond_to2pi: int=256) -> np.array:
    '''GS adapted for creeating holograms for moving traps
    '''

    w, l = target.shape
    target_amplitude = np.sqrt(target)
    space_norm = w * l
    norm = np.amax(target)
    error = tolerance + 1
    error_evolution = []
    n = 0
    A = ifft2(target_amplitude)
    while error > tolerance and n < max_loops:
        B = np.exp(1j * np.angle(A)) # our source amplitude is 1 everywhere
        C = fftshift(fft2(B))
        D = np.abs(target_amplitude) * np.exp(1j * np.angle(C))
        A = (ifft2(ifftshift(D)))
        expected_outcome = np.abs(C) ** 2
        expected_outcome *= norm / expected_outcome.max()
        error = error_f(expected_outcome, target, space_norm)
        error_evolution.append(error)
        n+=1
        if n % 10 == 0: print("-", end='')
    print()
    phase_for_slm = (np.angle(A) + np.pi) * correspond_to2pi / (2*np.pi) # converts phase to color value, input for SLM
    return phase_for_slm, expected_outcome, error_evolution

# def GS_without_fftshift(target: np.array, tolerance: float, max_loops: int, plot_error: bool=False, correspond_to2pi: int=256) -> np.array:
#     '''classical Gerchberg-Saxton algorithm
#     produces input for SLM for creating 'target' image
#     '''

#     w, l = target.shape
#     space_norm = w * l
#     error = tolerance + 1
#     error_evolution = []
#     n = 0
#     A = ifft2(target)
#     while error > tolerance and n < max_loops:
#         n+=1
#         B = A/abs(A) # our source amplitude is 1 everywhere
#         C = fft2(B)
#         D = np.abs(target) * C/abs(C)
#         A = ifft2(D)
#         expected_outcome = np.abs(C) # np.abs(fftshift(fft2(A/abs(A))))
#         scale = np.sqrt(255)/expected_outcome.max()
#         expected_outcome *= scale
#         error = error_f(expected_outcome**2, target**2, space_norm)
#         error_evolution.append(error)
#         if n % 10 == 0: print("-", end='')
#     print()
#     printout(error, n, error_evolution, "asdf", plot_error)
#     phase_for_slm = np.angle(A) * correspond_to2pi / (2*np.pi) # converts phase to color value, input for SLM
#     expected_outcome = expected_outcome**2
#     exp_tar_for_slm = expected_outcome * 255/np.amax(expected_outcome) # what the outcome from SLM should look like
#     return phase_for_slm, exp_tar_for_slm


def GD(demanded_output: np.array, args) -> Tuple[np.array, np.array, int]:
    
    incomming_intensity = np.array(im.open(args.path_to_incomming_intensity))
    incomming_amplitude = np.sqrt(incomming_intensity)
    w, l = demanded_output.shape
    space_norm = w * l
    initial_input = generate_initial_input(l, w)
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = args.tolerance + 1
    i = 0
    mask = 1 + args.mask_relevance * demanded_output/255
    print("computing hologram (one bar for 10 loops)", ' ')
    while error > args.tolerance and i < args.max_loops:
        med_output = fft2(input/abs(input) * incomming_amplitude)
        output_unnormed = abs(med_output) **2
        output = output_unnormed * norm / np.amax(output_unnormed) # toto prip. zapocitat do grad. zostupu
        dEdF = ifft2(mask * med_output * (output - demanded_output)) * incomming_amplitude
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= args.learning_rate * dEdX
        error = error_f(output, demanded_output, space_norm)
        error_evolution.append(error)
        if args.gif and i % args.gif_skip == 0:
            if args.gif_type == 'h':
                img = im.fromarray(complex_to_real_phase(input/abs(input), args.correspond_to2pi))
            if args.gif_type == 'i':
                img = im.fromarray(output)
            img.convert("RGB").save(f"{args.gif_source_address}/{i // args.gif_skip}.png")
        i += 1
        if args.unsettle and i % int(args.max_loops / args.unsettle) == 0:
            learning_rate *= 2
        if i % 10 == 0: print("-", end='')
    print()
    printout(error, i, error_evolution, f"learning_rate: {args.learning_rate}", args.plot_error)
    phase_for_slm = complex_to_real_phase(input, args.correspond_to2pi)
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

def complex_to_real_phase(complex_phase, correspond_to2pi=256):
    return (np.angle(complex_phase) + np.pi) / (2*np.pi) * correspond_to2pi

def dEdX_complex(dEdF, x):
    rE, iE = dEdF.real, dEdF.imag
    rx, ix = x.real, x.imag
    ax = abs(x)
    re_res = rE * (1/ax - rx**2 / ax**3) + iE * (- (rx*ix) / ax**3)
    im_res = rE * (- (rx*ix) / ax**3) + iE * (1/ax - ix**2 / ax**3)
    return re_res + 1j * im_res

def GD_for_moving_traps(demanded_output: np.array, initial_input: np.array, learning_rate: float=0.005,
       mask_relevance: float=10, tolerance: float=0.001, max_loops: int=50, unsettle=0, correspond_to2pi: int=256):
    w, l = demanded_output.shape
    space_norm = w * l
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    mask = 1 + mask_relevance * demanded_output/255
    i = 0
    while error > tolerance and i < max_loops:
        med_output = fft2(input/abs(input))
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(output_unnormed) * norm**2 # toto prip. zapocitat do grad. zostupu
        dEdF = ifft2(mask * med_output * (output - demanded_output))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = error_f(output, demanded_output, space_norm)
        error_evolution.append(error)
        i += 1
        if unsettle and i % int(max_loops / unsettle) == 0:
            learning_rate *= 2
        if i % 10 == 0: print("-", end='')
    print()
    phase_for_slm = complex_to_real_phase(input, correspond_to2pi)
    exp_tar_for_slm = output
    init_input_for_next = input
    return phase_for_slm, init_input_for_next, exp_tar_for_slm, error_evolution