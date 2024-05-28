# import explore_calibration as e
# import calibration_lib as cl
import numpy as np
import argparse
import time
import PIL.Image as im
import constants as c

def main(args):
    angle_e = args.angle.split("_")
    start = time.time()
    sample_cl = make_sample_holograms_cl(args.angle, args.precision, args.correspond_to2pi)
    after_cl = time.time()
    sample_e = make_sample_holograms_e(angle_e, args.precision, args.correspond_to2pi)
    after_e = time.time()
    print(f"in calibration_lib: {after_cl - start}, in explore_calibration: {after_e - after_cl}")
    for i in range(len(sample_cl)):
        difference = sample_cl[i] - sample_e[i]
        im.fromarray(difference).show()


def make_sample_holograms_cl(angle, precision, ct2pi):
    angle = angle.split("_")
    sample_holograms = []
    for i in range(precision):
        sample_holograms.append(decline(angle, i * 256 // precision, ct2pi))
    return sample_holograms

def make_sample_holograms_e(angle, precision, ct2pi):
    sample = []
    sample.append(decline(angle, 0, ct2pi))
    for i in range(1, precision):
        offset = i * 256 // precision
        sample.append((sample[0] + offset) % 256)
    return sample

def decline(angle, offset, ct2pi):
    x_angle, y_angle = angle
    hologram = np.zeros((c.slm_height, c.slm_width))
    const = ct2pi * c.px_distance / c.wavelength
    for i in range(c.slm_height):
        for j in range(c.slm_width):
            new_phase = const * (np.sin(float(y_angle) * c.u) * i + np.sin(float(x_angle) * c.u) * j)
            hologram[i, j] = int((new_phase + offset) % ct2pi)
    return hologram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--angle", type=str, default="1_1")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, default=256)
    args = parser.parse_args()
    main(args)