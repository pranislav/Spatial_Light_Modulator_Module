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
target = np.array(target_img)

initial_input = np.array([[(random() + 1j*random())\
     for _ in range(len(target[0]))] for _ in range(len(target))])



def GD_for_hologram_2D(initial_input: np.array, demanded_output: np.array,\
        learning_rate: float, tolerance: float):
    error_evolution = []
    norm = np.amax(demanded_output)
    input = initial_input
    error = tolerance + 1
    i = 0
    while error > tolerance:
        med_output = fft2(input/abs(input))
        output = abs(med_output) **2
        output = output / np.amax(abs(output)) * norm # toto prip. zapocitat do grad. zostupu
        dEdF = ifft2(med_output * (output - demanded_output))
        dEdX = np.array(list(map(dEdX_complex, dEdF, input)))
        input -= learning_rate * dEdX
        error = np.sum((output - demanded_output)**2, axis=(0, 1)) / (len(input) * len(input[0]))
        print(error)
        error_evolution.append(error)
        i += 1
        if i%30 == 0: learning_rate *= 2
    return input/abs(input), output, error_evolution


def dEdX_complex(dEdF, x):
    rE, iE = dEdF.real, dEdF.imag
    rx, ix = x.real, x.imag
    ax = abs(x)
    re_res = rE * (1/ax - rx**2 / ax**3) + iE * (- (rx*ix) / ax**3)
    im_res = rE * (- (rx*ix) / ax**3) + iE * (1/ax - ix**2 / ax**3)
    return re_res + 1j * im_res

def array_to_img(arr):
    return im.fromarray((np.angle(arr) + np.pi) / (2*np.pi) * 255)

learning_rate = 0.005
right_input, its_output, error_evolution = GD_for_hologram_2D(initial_input, target, learning_rate, 40)
input_phase = array_to_img(right_input)
# im.fromarray(its_output).show()

# input_phase.show()

plt.plot(error_evolution, label=str(learning_rate))
plt.legend()
plt.show()