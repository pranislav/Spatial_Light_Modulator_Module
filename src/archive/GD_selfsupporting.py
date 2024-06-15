import numpy as np
from scipy.fft import fft2, ifft2
from PIL import Image as im
import PIL.ImageOps
from random import random
import matplotlib.pyplot as plt



# name and type of image which should be projected by SLM
target_name = "smile"
target_type = "png"
invert = True

# loading image
target_img = im.open(f"images/{target_name}.{target_type}").convert('L').resize((1024, 768))
if invert:
    target_img = PIL.ImageOps.invert(target_img)
target = np.sqrt(np.array(target_img))



def GD_for_hologram_2D(demanded_output: np.array,learning_rate: float, \
                       tolerance: float, plot_error: bool=False):
    initial_input = generate_initial_input(len(demanded_output[0]), len(demanded_output))
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance:
        med_output = fft2(input/abs(input))
        output_unnormed = abs(med_output) **2
        output = output_unnormed / np.amax(abs(output_unnormed)) * norm # toto prip. zapocitat do grad. zostupu
        dEdF = ifft2(med_output * (output - demanded_output))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = np.sum((output - demanded_output)**2, axis=(0, 1)) / (len(input) * len(input[0]))
        print(error)
        error_evolution.append(error)
        i += 1
        # if i%30 == 0: learning_rate *= 2
    if plot_error:
        error_plot(error_evolution)
    phase_for_slm = complex_to_real_phase(input/abs(input))
    exp_tar_for_slm = output_unnormed / np.amax(abs(output_unnormed)) * norm**2 
    return phase_for_slm, exp_tar_for_slm


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

# def array_to_img(arr):
#     return im.fromarray((np.angle(arr) + np.pi) / (2*np.pi) * 255)

def error_plot(error_evolution: list):
    plt.plot(error_evolution, label=str(learning_rate))
    plt.legend()
    plt.show()


learning_rate = 0.05
source_phase_array, exp_tar_array = GD_for_hologram_2D(target, learning_rate, 0.5, plot_error=True)
source_phase = im.fromarray(source_phase_array)
im.fromarray(exp_tar_array).show()

source_phase.show()

