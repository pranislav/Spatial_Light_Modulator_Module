import argparse
import numpy as np
from PIL import Image as im
import wavefront_correction_lib as wcl
import phase_mask_smoothing as pms

def main(args):
    phase_mask = np.array(im.open(f"{args.source_dir}/{args.phase_mask_name}"))
    phase_mask = wcl.shrink_phase_mask(phase_mask, args.subdomain_size)
    unwrapped_mask = pms.unwrap_phase_picture(phase_mask, args.correspond_to_2pi)
    corrected_mask = wcl.fit_and_subtract(unwrapped_mask, wcl.quadratic_func, [0, 0])
    upscaled_mask = im.fromarray(corrected_mask % args.correspond_to2pi).resize((wcl.slm_width, wcl.slm_height), resample=im.BILINEAR)
    upscaled_mask.convert("L").save(f"{args.source_dir}/{args.phase_mask_name[:-4]}_removed_defocus.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source_dir = "lc-slm/holograms/wavefront_correction_phase_masks"
    parser.add_argument("phase_mask_name", type=str, help=f"phase mask to remove defocus from {source_dir}")
    parser.add_argument("-s", "--subdomain_size", type=int, default=32, help="subdomain size used to create the phase mask")
    parser.add_argument("-ct2pi", "--correspond_to_2pi", type=int, default=245, help="value of pixel corresponding to 2pi phase shift")
    args = parser.parse_args()
    args.source_dir = source_dir
    main(args)
