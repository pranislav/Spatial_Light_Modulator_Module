import argparse
from PIL import Image as im
import numpy as np
import help_messages_wfc

def show_hologram(args):
    hologram = np.load(args.hologram)
    hologram_int = (hologram % (2 * np.pi) * args.correspond_to2pi / (2 * np.pi)).astype(np.uint8)
    hologram_img = im.fromarray(hologram_int)
    hologram_img.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hologram", help="address of the hologram")
    parser.add_argument("correspond_to2pi", type=int, default=255, help=help_messages_wfc.ct2pi)
    args = parser.parse_args()
    show_hologram(args)
