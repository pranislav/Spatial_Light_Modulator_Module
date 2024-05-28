import explore_calibration as e
import calibration_lib as cl
import numpy as np
import argparse
import time
import PIL.Image as im

def main(args):
    angle_e = args.angle.split("_")
    start = time.time()
    sample_cl = cl.make_sample_holograms(args.angle, args.precision, args.correspond_to2pi)
    after_cl = time.time()
    sample_e = e.make_sample_holograms(angle_e, args.precision, args.correspond_to2pi)
    after_e = time.time()
    print(f"cl: {after_cl - start}, e: {after_e - after_cl}")
    for i in range(len(sample_cl)):
        difference = sample_cl[i] - sample_e[i]
        im.fromarray(difference).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--angle", type=str, default="1_1")
    parser.add_argument("-p", "--precision", type=int, default=2)
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, default=256)
    args = parser.parse_args()
    main(args)