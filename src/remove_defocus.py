import argparse
import numpy as np
from PIL import Image as im
import wavefront_correction_lib as wcl
import phase_mask_smoothing as pms
import constants as c
import help_messages_wfc
import os


def main(args):
    phase_mask = np.array(im.open(f"{args.source_dir}/{args.phase_mask_name}"))
    phase_mask = pms.shrink_phase_mask(phase_mask, args.subdomain_size)
    unwrapped_mask = pms.unwrap_phase_picture(phase_mask, args.correspond_to2pi)
    corrected_mask = wcl.fit_and_subtract(unwrapped_mask, wcl.quadratic_func, [0, 0])
    upscaled_mask = im.fromarray(corrected_mask % args.correspond_to2pi).resize((c.slm_width, c.slm_height), resample=im.BILINEAR)
    base, ext = os.path.splitext(args.phase_mask_name)
    upscaled_mask.convert("L").save(f"{args.source_dir}/{base}_removed_defocus{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source_dir = "holograms/wavefront_correction_phase_masks"
    parser.add_argument("phase_mask_name", type=str, help=f"phase mask to remove defocus from {source_dir}")
    parser.add_argument("-s", "--subdomain_size", metavar="INT", type=int, default=32, help="subdomain size used to create the phase mask")
    parser.add_argument("-ct2pi", "--correspond_to2pi", metavar="INT", type=int, required=True, help=help_messages_wfc.ct2pi)
    args = parser.parse_args()
    args.source_dir = source_dir
    main(args)
