import slm_screen as ss
import PIL.Image as im
import constants as c
import calibration_lib as cl
import argparse

hologram = ss.Screen()

hologram.decline("x", 1 * c.u).decline("y", 1 * c.u)

hologram.img.save("lc-slm/holograms/analytical/decline_x1u_y1u.png")

# hologram.lens(0.6)
# hologram.img.save("holograms/analytical_xdecline_2u.jpg")

def create_decline_hologram(args):
    angle = args.angle.split("_")
    hologram = cl.decline(angle, 0, args.correspond_to2pi)
    im.fromarray(hologram).convert("L").save(f"{args.directory}/decline_{args.angle}_ct2pi_{args.correspond_to2pi}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ct2pi", "--correspond_to2pi", default=256, type=int, help="color value corresponding to 2pi phase change on SLM")
    parser.add_argument("-d", "--directory", default="lc-slm/holograms/analytical", type=str, help="directory to save the hologram")
    parser.add_argument("-a", "--angle", default="1_1", type=str, help="angle of the decline. use form: xdecline_ydecline (angles in constants.u unit)")
    args = parser.parse_args()
    create_decline_hologram(args)