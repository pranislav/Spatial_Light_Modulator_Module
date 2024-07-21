import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import random
import PIL.Image as im
from typing import Tuple
from cmath import phase


def GS(demanded_output, args):
    '''classical Gerchberg-Saxton algorithm
    produces input for SLM for creating 'demanded_output' image
    '''

    incomming_intensity = np.ones(demanded_output.shape) if args.incomming_intensity == "uniform" else  np.array(im.open(args.incomming_intensity))
    incomming_amplitude = np.sqrt(incomming_intensity)
    w, l = demanded_output.shape
    demanded_output_amplitude = np.sqrt(demanded_output)
    space_norm = w * l
    norm = np.amax(demanded_output)
    error = args.tolerance + 1
    error_evolution = []
    i = 0
    A = ifft2(demanded_output_amplitude) # A = ifft2(ifftshift(demanded_output_amplitude))
    while error > args.tolerance and i < args.max_loops:
        B = incomming_amplitude * np.exp(1j * np.angle(A))
        C = fft2(B) # C = fftshift(fft2(B))
        D = np.abs(demanded_output_amplitude) * np.exp(1j * np.angle(C))
        A = ifft2(D) # A = ifft2(ifftshift(D))
        expected_outcome = np.abs(C)**2
        expected_outcome *= norm / expected_outcome.max()
        error = error_f(expected_outcome, demanded_output, space_norm)
        error_evolution.append(error)
        if args.gif and i % args.gif_skip == 0:
            add_gif_image(args, expected_outcome, A, i)
        i += 1
        print(f"\rloop {i}/{args.max_loops}", end='')
    print()
    if args.print_info:
        print()
        printout(error, i, error_evolution, args.plot_error)
    phase_for_slm = (np.angle(A) + np.pi) * args.correspond_to2pi / (2*np.pi) # converts phase to color value, input for SLM
    return phase_for_slm, expected_outcome, error_evolution


def add_gif_image(args, expected_outcome, A, i):
    if args.gif_type == 'h':
        img = im.fromarray((np.angle(A) + np.pi) * args.correspond_to2pi / (2*np.pi))
    if args.gif_type == 'i':
        img = im.fromarray(expected_outcome)
    img.convert("L").save(f"{args.gif_source_address}/{i // args.gif_skip}.png")


def GD(demanded_output: np.array, args):
    
    incomming_intensity = np.ones(demanded_output.shape) if args.incomming_intensity == "uniform" else  np.array(im.open(args.incomming_intensity))
    incomming_amplitude = np.sqrt(incomming_intensity)
    w, l = demanded_output.shape
    space_norm = w * l
    error_evolution = []
    norm = np.amax(demanded_output)
    input = make_initial_guess(args.initial_guess, incomming_amplitude, demanded_output, args.random_seed)
    error = args.tolerance + 1
    i = 0
    mask = 1 + args.white_attention * demanded_output/255
    if args.print_info: print("computing hologram")
    while error > args.tolerance and i < args.max_loops:
        med_output = fft2(input/abs(input) * incomming_amplitude)
        output_unnormed = abs(med_output) **2
        output = output_unnormed * norm / np.amax(output_unnormed) # toto prip. zapocitat do grad. zostupu
        # im.fromarray(output).show()
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
            img.convert("L").save(f"{args.gif_source_address}/{i // args.gif_skip}.png")
        i += 1
        if args.unsettle and i % int(round(args.max_loops / (args.unsettle + 1))) == 0:
            args.learning_rate *= 2
        print(f"\rloop {i}/{args.max_loops}", end='')
    print()
    if args.print_info:
        print()
        printout(error, i, error_evolution, args.plot_error)
    phase_for_slm = complex_to_real_phase(input, args.correspond_to2pi)
    exp_tar_for_slm = output
    return phase_for_slm, exp_tar_for_slm, error_evolution


def make_initial_guess(initial_guess_type, incomming_amplitude, demanded_output, seed):
    h, w = demanded_output.shape
    random.seed(seed)
    if initial_guess_type == "random":
        return np.array([[np.exp(1j * 2 * np.pi * random.random()) for _ in range(w)] for _ in range(h)]) 
    if initial_guess_type == "old":
        return np.array([[(np.sqrt(random.random()) + 1j*np.sqrt(random.random())) for _ in range(w)] for _ in range(h)])
    if initial_guess_type == "unnormed":
        return (np.array([[random.random() + 1j*random.random() for _ in range(w)] for _ in range(h)]) - 0.5) * 2
    if initial_guess_type == "zeros":
        return np.array([[np.exp(1j * 2 * np.pi * random.random()) / 100 for _ in range(w)] for _ in range(h)]) 
    if initial_guess_type == "ones":
        return np.ones(demanded_output.shape) + 1j * np.zeros(demanded_output.shape)
    elif initial_guess_type == "fourier":
        return incomming_amplitude * np.exp(1j * np.angle(ifft2(np.sqrt(demanded_output))))
    else:
        raise ValueError("unknown type of initial guess")

def error_f(actual, correct, norm):
    return np.sum((actual - correct)**2) / norm


def printout(error, loop_num, error_evol, plot_error):
    print(f"error: {error}")
    print(f"number of loops: {loop_num}")
    if plot_error:
        plt.plot(error_evol)
        plt.xlabel("loop number")
        plt.ylabel("error")
        plt.show()


def complex_to_real_phase(complex_phase, correspond_to2pi=256):
    return (np.angle(complex_phase) + np.pi) / (2*np.pi) * correspond_to2pi

def dEdX_complex(dEdF, x):
    rE, iE = dEdF.real, dEdF.imag
    rx, ix = x.real, x.imag
    ax = abs(x)
    re_res = rE * (1/ax - rx**2 / ax**3) + iE * (- (rx*ix) / ax**3)
    im_res = rE * (- (rx*ix) / ax**3) + iE * (1/ax - ix**2 / ax**3)
    return re_res + 1j * im_res
