import numpy as np
from PIL import Image as im
from scipy.fft import fft2, ifft2
from time import sleep

i = 1j

def GS(target: np.array, accuracy: float) -> np.array:
    A = ifft2(target)
    error = accuracy + 1
    n = 0
    while error > accuracy:
        n+=1
        B = np.exp(i * np.angle(A)) # our source amplitude is 1 everywhere
        C = fft2(B)
        D = np.abs(target) * np.exp(i * np.angle(C))
        A = ifft2(D)
        exp_tar = np.abs(expected_target(np.angle(A)))
        error = error_fun(exp_tar, target)
        if n > 200:
            break
    print(error)
    print(n)
    phase_for_slm = np.angle(A) * 255 / (2*np.pi)
    exp_tar_for_slm = exp_tar * 255/max(exp_tar.max(0))
    return phase_for_slm, exp_tar_for_slm


def expected_target(source_phase: np.array) -> np.array:
    return fft2(np.exp(i * source_phase))


def error_fun(exp_tar: np.array, target: np.array) -> float:
    # im.fromarray(exp_tar).show()
    # sleep(2)
    # im.fromarray(exp_tar*255/max(exp_tar.max(0))).show()
    # # im.fromarray(target).show()
    # sleep(2)

    error_vec = target - exp_tar*np.sqrt(255)/max(exp_tar.max(0))
    error = sum(sum(error_vec))
    # print(abs(error))
    return abs(error)




