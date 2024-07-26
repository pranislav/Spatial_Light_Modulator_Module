import numpy as np
import time
import constants as c

def measure_hologram_conversion_to_ints(hologram, correspond_to2pi):
    start = time.time()
    int_hologram = (hologram * correspond_to2pi / (2 * np.pi)).astype(np.uint8)
    print("Time taken to convert hologram to ints:", time.time() - start)
    return int_hologram

def main():
    hologram = np.random.rand(c.slm_height, c.slm_width)
    correspond_to2pi = 255
    hologram = measure_hologram_conversion_to_ints(hologram, correspond_to2pi)

if __name__ == "__main__":
    main()
