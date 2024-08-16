from algorithms import gerchberg_saxton, gradient_descent
import numpy as np
from PIL import Image as im
import PIL.ImageOps
import constants as c
import argparse
import os
import imageio
import wavefront_correction as wfc
from scipy.fft import fft2


def main(args):
    if args.img_name is None:
        hologram = np.zeros((c.slm_height, c.slm_width))
    else:
        hologram = make_hologram(args)
    hologram = transform_hologram(hologram, args)
    if args.preview:
        show_expected_outcome(hologram, args)
    save_hologram_and_gif(hologram, args)


def show_expected_outcome(hologram, args):
    expected_outcome = fft2(np.exp(1j * hologram))
    norm = find_out_norm(args)
    expected_outcome_intensity = np.abs(expected_outcome) ** 2
    expected_outcome_normed = (
        expected_outcome_intensity / np.amax(expected_outcome_intensity) * norm
    )
    square_prev_image = im.fromarray(expected_outcome_normed).resize(
        (c.slm_height, c.slm_height)
    )
    square_prev_image.show()


def find_out_norm(args):
    if args.img_name is None:
        return 255
    img = im.open(f"images/{args.img_name}").convert("L")
    img_arr = np.array(img)
    return np.amax(img_arr)


def pad_to_square(img):
    """Pads the image with black pixels to make it square"""

    # Get current dimensions
    width, height = img.size

    if width == height:
        return img

    # Determine the size of the new square image
    new_size = max(width, height)

    # Create a new black image with square dimensions in 'L' mode
    new_img = im.new("L", (new_size, new_size), 0)

    # Calculate position to paste the original image
    paste_x = (new_size - width) // 2
    paste_y = (new_size - height) // 2

    # Paste the original grayscale image onto the new black image
    new_img.paste(img, (paste_x, paste_y))

    return new_img


def make_hologram(args):
    algorithm = (
        gerchberg_saxton if args.algorithm == "gerchberg_saxton" else gradient_descent
    )
    target = prepare_target(args.img_name, args)
    if args.gif:
        add_gif_dirs(args)
        remove_files_in_dir(args.gif_source_dir)
    hologram, _, _ = algorithm(target, args)
    return hologram


def transform_hologram(hologram, args):
    if args.deflect is not None:
        hologram = deflect_hologram(hologram, args.deflect)
    if args.lens:
        hologram = add_lens(hologram, args.lens)
    return hologram


def add_gif_dirs(args):
    if args.gif_type == "h":
        args.gif_dest_dir = "holograms"
    elif args.gif_type == "i":
        args.gif_dest_dir = "images"
    if not os.path.exists(args.gif_dest_dir):
        os.makedirs(args.gif_dest_dir)
    args.gif_source_dir = f"{args.gif_dest_dir}/gif_source"
    if not os.path.exists(args.gif_source_dir):
        os.makedirs(args.gif_source_dir)


def prepare_target(img_name, args):
    target_img = im.open(f"images/{img_name}").convert("L")
    if args.invert:
        target_img = PIL.ImageOps.invert(target_img)
    target_img = pad_to_square(target_img)
    if args.quarterize:
        target_img = quarter(target_img)
    resized = target_img.resize((int(c.slm_width), int(c.slm_height)))
    return np.array(resized)


def save_hologram_and_gif(hologram, args):
    img_name = (
        os.path.basename(args.img_name).split(".")[0] if args.img_name else "analytical"
    )
    dest_dir = args.destination_directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    hologram_name = make_hologram_name(args, img_name)
    hologram_name_orig = wfc.originalize_name(f"{dest_dir}/{hologram_name}.npy")
    np.save(hologram_name_orig, hologram)
    # im.fromarray(hologram).convert("L").save(hologram_name_img)
    if args.gif:
        hologram_name_gif = wfc.originalize_name(
            f"{args.gif_dest_dir}/{hologram_name}.gif"
        )
        create_gif(args.gif_source_dir, hologram_name_gif)


def make_hologram_name(args, img_name):
    alg_params = ""
    transforms = ""
    img_transforms = ""
    if args.deflect:
        transforms += f"_deflect_x{args.deflect[0]}_y{args.deflect[1]}"
    if args.lens:
        transforms += f"_lens{args.lens}"
    if args.algorithm == "gradient_descent":
        alg_params += (
            f"_lr{args.learning_rate}_mr{args.white_attention}_unsettle{args.unsettle}"
        )
    if args.quarterize:
        img_transforms += "_quarter"
    if args.invert:
        img_transforms += "_inverted"
    if args.img_name is None:
        return f"{img_name}{transforms}"
    return f"{img_name}{img_transforms}\
        _{args.algorithm}\
        {alg_params}\
        _loops{args.max_loops}\
        {transforms}"


def args_to_string(args):
    arg_string = ""
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            arg_string += f"{arg_name}_{arg_value}_"
    # Remove the trailing underscore
    arg_string = arg_string.rstrip("_")
    return arg_string


def quarter(image: im) -> im:
    """returns mostly blank image with original image pasted in upper-left corner
    when generated hologram for such a transformed image, there will be no overlap
    between different diffraction order of displayed image
    """
    w, h = image.size
    resized = image.resize((w // 2, h // 2))
    ground = im.new("L", (w, h))
    ground.paste(resized)
    return ground


def deflect_hologram(hologram: np.array, angle: tuple):
    """deflects hologram by angle, returns deflectd hologram"""
    deflect = wfc.deflect_2pi(angle)
    deflected_hologram = (hologram + deflect) % (2 * np.pi)
    return deflected_hologram


def add_lens(hologram: np.array, focal_len: float):
    return (hologram + lens(focal_len, hologram.shape)) % (2 * np.pi)


def lens(focal_length, shape):
    """simulates lens with focal length 'focal_length' in meters"""
    h, w = shape
    hologram = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            r = c.px_distance * np.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            phase_shift = (
                2 * np.pi
                * focal_length
                / c.wavelength
                * (1 - np.sqrt(1 + r**2 / focal_length**2))
            )
            hologram[i, j] = phase_shift % (2 * np.pi)
    return hologram


def create_gif(img_dir, outgif_path):
    """creates gif from images in img_dir
    and saves it as outgif_path
    """
    with imageio.get_writer(outgif_path, mode="I") as writer:
        for file in os.listdir(img_dir):
            image = imageio.imread(f"{img_dir}/{file}")
            writer.append_data(image)


def remove_files_in_dir(dir_name):
    """removes all files in a directory"""
    for file in os.listdir(dir_name):
        os.remove(f"{dir_name}/{file}")


if __name__ == "__main__":
    description = """Generate phase hologram for transmissive SLM.
    When displayed on SLM, the hologram will create image on the Fourier plane.
    Provide path to image to be reconstructed or leave empty to create pure deflect/lens hologram.
    The image should be in images directory.
    Image is padded with black pixels to make it square so proportions of the reconstructed image are preserved.
    Given image plus specified deflection results in image shifted by given angle on the Fourier plane.
    Holograms are saved in holograms directory as .npy files.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description
    )
    parser.add_argument(
        "img_name",
        nargs="?",
        default=None,
        type=str,
        help="path to the target image from images directory. Leave empty if you want to create pure deflect/lens hologram",
    )
    parser.add_argument(
        "-ii",
        "--incomming_intensity",
        type=str,
        default="uniform",
        help="path to the incomming intensity image from images directory or 'uniform' for uniform intensity",
    )
    parser.add_argument(
        "-ig",
        "--initial_guess",
        type=str,
        default="random",
        choices=["random", "fourier"],
        help="initial guess for the gradient_descent algorithm: random or phase from the Fourier transform of the target image",
    )
    parser.add_argument(
        "-dest_dir",
        "--destination_directory",
        type=str,
        default="holograms",
        help="directory where the hologram will be saved",
    )
    parser.add_argument(
        "-q",
        "--quarterize",
        action="store_true",
        help="the original image is pasted to one quadrant of a black image\
            before passed to hologram generating to make sure there\
            will be no overlaps between various aliases of displayed image.",
    )
    parser.add_argument(
        "-i",
        "--invert",
        action="store_true",
        help="invert colors of the target image"
    )
    parser.add_argument(
        "-alg",
        "--algorithm",
        default="gerchberg_saxton",
        choices=["gerchberg_saxton", "gradient_descent"],
        help="algorithm to use for hologram generation",
    )
    parser.add_argument(
        "-tol",
        "--tolerance",
        default=0,
        metavar="FLOAT",
        type=float,
        help="algorithm stops when error descends under tolerance",
    )
    parser.add_argument(
        "-l",
        "--max_loops",
        default=42,
        metavar="INTEGER",
        type=int,
        help="algorithm performs no more than max_loops loops no matter what error it is",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.005,
        type=float,
        help="learning rate for gradient descent algorithm (how far the solution jumps in direction of the gradient)",
    )
    parser.add_argument(
        "-wa",
        "--white_attention",
        metavar="FLOAT",
        default=1,
        type=float,
        help="attention to white places for gradient_descent algorithm, sets higher priority to white areas by making error on those areas white_attention-times higher",
    )
    parser.add_argument(
        "-u",
        "--unsettle",
        default=0,
        metavar="INTEGER",
        type=int,
        help="unsettle for gradient descent algorithm; learning rate is unsettle times doubled",
    )
    parser.add_argument(
        "-gif",
        action="store_true",
        help="create gif from hologram computing evolution"
    )
    parser.add_argument(
        "-gif_t",
        "--gif_type",
        choices=["h", "i"],
        default="i",
        help="type of gif: h for hologram, i for image (result)",
    )
    parser.add_argument(
        "-gif_skip",
        default=1,
        type=int,
        metavar="INTEGER",
        help="each gif_skip-th frame will be in gif",
    )
    parser.add_argument(
        "-plot_error",
        action="store_true",
        help="plot error evolution"
    )
    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="show expected outcome at the end of the program run",
    )
    parser.add_argument(
        "-deflect",
        nargs=2,
        type=float,
        metavar=("X_ANGLE", "Y_ANGLE"),
        default=None,
        help="add hologram for deflect to computed hologram. Effect: shifts resulting image on Fourier plane by given angle (in units of quarter of first diffraction maximum)",
    )
    parser.add_argument(
        "-lens",
        default=None,
        type=float,
        metavar="FOCAL_LENGTH",
        help="add lens to hologram with given focal length in meters",
    )
    args = parser.parse_args()
    args.random_seed = 42
    args.print_info = True
    if not os.path.exists(args.destination_directory):
        os.makedirs(args.destination_directory)

    main(args)
