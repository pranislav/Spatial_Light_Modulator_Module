import calibration_lib as cl
import copy
from PIL import Image as im
import argparse

def is_equal_blazing_one_period_phase_shift(correspond_to2pi, angle):
    hologram = cl.decline(angle, correspond_to2pi)
    shifted_hologram = (copy.deepcopy(hologram) + correspond_to2pi) % correspond_to2pi
    # im.fromarray(hologram).show()
    im.fromarray(shifted_hologram).show()
    im.fromarray(hologram- shifted_hologram).show()
    return (hologram == shifted_hologram).all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ct2pi", "--correspond_to2pi", type=int, default=256)
    parser.add_argument("-a", "--angle", type=str, default="0.3_0.3")
    args = parser.parse_args()
    print(is_equal_blazing_one_period_phase_shift(args.correspond_to2pi, tuple(map(float, args.angle.split("_")))))