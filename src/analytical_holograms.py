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
    w, h = shape
    hologram = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            r = c.px_distance * np.sqrt((i - w / 2) ** 2 + (j - h / 2) ** 2)
            phase_shift = 2 * np.pi * focal_length / c.wavelength * \
                (1 - np.sqrt(1 + r ** 2 / focal_length ** 2))
            hologram[i, j] = (phase_shift * correspond_to2pi / (2 * np.pi)) % correspond_to2pi
    return hologram

def create_decline_hologram(args):
    angle = cl.read_angle(args.angle)
    hologram = cl.decline(angle, args.correspond_to2pi)
    im.fromarray(hologram).convert("L").save(f"{args.directory}/decline_{args.angle}_ct2pi_{args.correspond_to2pi}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ct2pi", "--correspond_to2pi", default=256, type=int, help="color value corresponding to 2pi phase change on SLM")
    parser.add_argument("-d", "--directory", default="holograms/analytical", type=str, help="directory to save the hologram")
    parser.add_argument("-a", "--angle", default="1_1", type=str, help="angle of the decline. use form: xdecline_ydecline (angles in constants.u unit)")
    args = parser.parse_args()
    create_decline_hologram(args)