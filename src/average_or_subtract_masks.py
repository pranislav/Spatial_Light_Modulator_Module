import argparse
import numpy as np
from PIL import Image as im
import phase_mask_smoothing as pms
from skimage.restoration import unwrap_phase
import constants as c
import wavefront_correction as wfc


def main(args):
    masks = [np.load(f"{source_dir}/{image}") for image in args.images]
    if args.subtract:
        subtract_mask = np.load(f"{source_dir}/{args.subtract}")
    average_mask = np.zeros(masks[0].shape)
    for mask in masks:
        average_mask += unwrap_phase(mask - np.pi)
    average_mask /= len(masks)
    if args.subtract:
        average_mask -= unwrap_phase(subtract_mask - np.pi)
    average_mask = wfc.resize_2d_array(average_mask, (c.slm_height, c.slm_width))
    average_mask_wrapped = average_mask % (2 * np.pi)
    average_mask_resized = wfc.resize_2d_array(average_mask_wrapped, (c.slm_height, c.slm_width))
    save_name = args.source_dir + "/" + args.output_name
    np.save(save_name, average_mask_resized)
    int_mask = wfc.convert_2pi_hologram_to_int_hologram(average_mask_resized, args.correspond_to2pi)
    im.fromarray(int_mask).convert("L").save(save_name + ".png")
    print(f"Resulting mask saved as {save_name}.npy and {save_name}.png")


if __name__ == "__main__":
    source_dir = "holograms/wavefront_correction_phase_masks"
    description = f"""Average provided masks and subtract specified mask if given.
    Masks should be from directory {source_dir} and in .npy format.
    All of them should be of same type - smoothed or not smoothed.
    The result is saved to the source directory under output_name in both .npy and .png formats.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )
    parser.add_argument(
        "output_name",
        type=str,
        help="name of the average mask"
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help=f"list of images to average from {source_dir}",
    )
    parser.add_argument(
        "-subtract",
        metavar="PATH",
        type=str,
        help="image to subtract from the average"
    )
    parser.add_argument(
        "-ct2pi",
        "--correspond_to2pi",
        metavar="INT",
        type=int,
        default=256,
        help="value of pixel corresponding to 2pi phase shift",
    )
    args = parser.parse_args()
    args.source_dir = source_dir
    main(args)
