import algorithms as alg
from PIL import Image as im
import numpy as np
import constants as c
from matplotlib import pyplot as plt


def make_list_of_tuples_of_random_tuples(number_of_tuples):
    return [((np.random.randint(c.slm_height//2), np.random.randint(c.slm_width//2)), (np.random.randint(c.slm_height//2), np.random.randint(c.slm_width//2))) for _ in range(number_of_tuples)]

def make_two_trap_hologram(black_image, coords):
    black_image[coords[0][0]][coords[0][1]] = 255
    black_image[coords[1][0]][coords[1][1]] = 255
    hologram = np.angle(np.fft.ifft2(black_image))
    return hologram

def expected_image(hologram):
    image_amplitude = np.fft.fft2(np.exp(1j * hologram))
    image = np.abs(image_amplitude)**2
    image = (image / np.max(image) * 255).astype(np.uint8)
    return image

def check_hologram(coords):
    black_image = np.zeros((c.slm_height, c.slm_width), dtype=np.uint8)
    hologram = make_two_trap_hologram(black_image, coords)
    image = expected_image(hologram)
    error = alg.error_f(image, black_image, 1)
    print(f"Error: {error}")
    print(image[coords[0]], image[coords[1]])
    # plt.plot(image[42])
    # plt.show()
    # image = im.fromarray(image)
    # image.show()
    image[coords[0][0]][coords[0][1]] = 0
    image[coords[1][0]][coords[1][1]] = 0
    print(np.max(image))

def main():
    number_of_tuples = 10
    coords = make_list_of_tuples_of_random_tuples(number_of_tuples)
    for coord in coords:
        check_hologram(coord)

if __name__ == "__main__":
    main()
