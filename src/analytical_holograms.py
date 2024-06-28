# import slm_screen as ss
import PIL.Image as im
import constants as c
import wavefront_correction_lib as cl
import argparse
import numpy as np

# hologram = ss.Screen()

# hologram.decline("x", 1 * c.u).decline("y", 1 * c.u)

# hologram.img.save("holograms/analytical/decline_x1u_y1u.png")

# hologram.lens(0.6)
# hologram.img.save("holograms/analytical_xdecline_2u.jpg")

def lens(focal_length, correspond_to2pi, shape):
    '''simultes lens with focal length 'focal_length' in meters
    '''
    h, w = shape
    hologram = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r = c.px_distance * np.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            phase_shift = 2 * np.pi * focal_length / c.wavelength * \
                (1 - np.sqrt(1 + r ** 2 / focal_length ** 2))
            hologram[i, j] = (phase_shift * correspond_to2pi / (2 * np.pi)) % correspond_to2pi
    return hologram

def create_decline_hologram(args):
    hologram = cl.decline(args.decline, args.correspond_to2pi)
    im.fromarray(hologram).convert("L").save(f"{args.directory}/decline_{args.decline[0]}_{args.decline[1]}_ct2pi_{args.correspond_to2pi}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ct2pi", "--correspond_to2pi", default=256, type=int, help="color value corresponding to 2pi phase change on SLM")
    parser.add_argument("-dir", "--directory", default="holograms/analytical", type=str, help="directory to save the hologram")
    parser.add_argument("-d", "--decline", nargs=2, default=(0.5, 0.5), type=float, help="angle to decline the light in x and y direction (in constants.u unit)")
    args = parser.parse_args()
    create_decline_hologram(args)