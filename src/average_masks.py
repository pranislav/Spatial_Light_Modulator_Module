import argparse
import numpy as np
from PIL import Image as im
import phase_mask_smoothing as pms
from skimage.restoration import unwrap_phase
import constants as c


def main(args):
    masks = [np.array(im.open(f"{source_dir}/{image}")) for image in args.images]
    subtract_mask = np.array(im.open(f"{source_dir}/{args.subtract}")) if args.subtract else None
    if args.smooth:
        masks = [pms.shrink_phase_mask(mask, args.subdomain_size) for mask in masks]
        if subtract_mask is not None:
            subtract_mask = pms.shrink_phase_mask(subtract_mask, args.subdomain_size)
    average_mask = np.zeros(masks[0].shape)
    for mask in masks:
        average_mask += unwrap_phase(pms.transform_to_phase_values(mask, args.ct2pi))
    average_mask /= len(masks)
    if subtract_mask is not None:
        average_mask -= unwrap_phase(pms.transform_to_phase_values(subtract_mask, args.ct2pi))
    average_mask = pms.transform_to_color_values(average_mask, args.ct2pi)
    average_mask = im.fromarray(average_mask)
    if args.smooth:
        average_mask = average_mask.resize((c.slm_width, c.slm_height), resample=im.BILINEAR)
    average_mask = im.fromarray(np.array(average_mask) % args.ct2pi).convert("L")
    average_mask.save(f"{source_dir}/{args.output_name}.png")


if __name__ == "__main__":
    source_dir = "holograms/wavefront_correction_phase_masks"
    parser = argparse.ArgumentParser()
    parser.add_argument("output_name", type=str)
    parser.add_argument("images", type=str, nargs="+", help=f"list of images to average from {source_dir}")
    parser.add_argument("-subtract", type=str, help="image to subtract from the average")
    parser.add_argument("-ct2pi", type=int, default=245, help="value of pixel corresponding to 2pi phase shift")
    parser.add_argument("-s", "--smooth", action="store_true", help="resize with bilinear interpolation to the average mask")
    parser.add_argument("-ss", "--subdomain_size", type=int, default=32, help="subdomain size used to create the phase mask")
    args = parser.parse_args()
    main(args)
