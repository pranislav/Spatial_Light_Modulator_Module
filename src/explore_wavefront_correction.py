'''script which interacts with the experiment and provides
an insight into the experiment response to variaous parameters change
it displays a wavefront_correction hologram specified by user on the SLM
and then shows the hologram, an image from camera and
evolution of intensity with respect to phase shift of current subdomain
fitted with cosine and prints out fitted parameters

main goal is to help the user to estimate optimal configuration for
wavefront_correction and color-phase relation search

parameters to be changed:
- subdomain size
- position of subdomain
- position of reference subdomain
- declining angle
- nuber of phase shifts
- number of frames to average to supress the fluctuation
'''

# TODO: subdomain_position degeneration (real & input one)

import constants as c
import wavefront_correction_lib as cl
import display_holograms as dh
import numpy as np
from PIL import Image as im
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pylablib.devices import uc480
import argparse
import cv2
import os
from copy import deepcopy
import time
import fit_stuff as f


def explore(args):
    params = default_params()
    # sample_list = make_sample_holograms(sample_list, params)
    window = cl.create_tk_window()
    cam = uc480.UC480Camera()
    internal_screen_resolution = get_internal_screen_resolution()
    video_dir = "lc-slm/images/explore"
    if not os.path.exists(video_dir): os.makedirs(video_dir)
    while True:
        black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
        if params["decline"][-1] or params["precision"][-1] or params["correspond_to_2pi"][-1]:
            angle = last_nonempty(params["decline"])
            precision = last_nonempty(params["precision"])
            correspond_to_2pi = last_nonempty(params["correspond_to_2pi"])
            sample_list = cl.make_sample_holograms(angle, precision, correspond_to_2pi)
        if params["subdomain_size"][-1] or params["reference_position"][-1]:
            subdomain_size = last_nonempty(params["subdomain_size"])
            reference_position = real_subdomain_position(last_nonempty(params["reference_position"]), subdomain_size)
        reference_hologram = cl.add_subdomain(black_hologram, sample_list[0], reference_position, subdomain_size)
        num_to_avg = last_nonempty(params["num_to_avg"])
        if params["decline"][-1] or params["subdomain_size"][-1]:
            cl.set_exposure_wrt_reference_img(cam, window, (256 / 4 - 20, 256 / 4), reference_hologram, num_to_avg)
        if params["decline"][-1] or params["subdomain_size"][-1] or params["correspond_to_2pi"][-1]:
            intensity_coord = cl.get_highest_intensity_coordinates_img(cam, window, reference_hologram, num_to_avg)
        hologram = reference_hologram
        subdomain_position = real_subdomain_position(last_nonempty(params["subdomain_position"]), subdomain_size)
        frames, intensity_data = wavefront_correction_loop_explore(window, cam, hologram, sample_list, subdomain_position, subdomain_size, precision, intensity_coord, num_to_avg)
        fit_params = f.fit_intensity_general(intensity_data, f.positive_cos)
        print_fit_params(fit_params)
        intensity_fit = plot_fit(fit_params)
        frame_img_list = dot_frames(frames, intensity_coord)
        output, video_frame_info = format_output(hologram, frame_img_list[0], intensity_data, intensity_fit, internal_screen_resolution)
        if args.mode == 'i':
            output.show()
        elif args.mode == 'v':
            name = make_name(params)
            img_list = make_imgs_for_video(output, frame_img_list, video_frame_info)
            images_to_video(img_list, name, 3, video_dir) # TODO: treat fps in other way

        if input("continue (enter) or quit (anything) >> "): break
        get_params(params)


def wavefront_correction_loop_explore(window, cam, hologram, sample, subdomain_position, subdomain_size, precision, coordinates, num_to_avg):
    intensity_list = [[], []]
    images_list = []
    k = 0
    while k < precision:
        hologram = cl.add_subdomain(hologram, sample[k], subdomain_position, subdomain_size)
        cl.display_image_on_external_screen(window, hologram)
        # time.sleep(0.1)
        intensity = 0
        for _ in range(num_to_avg):
            frame = cam.snap()
            intensity += cl.get_intensity_on_coordinates(frame, coordinates)
        images_list.append(frame)
        intensity /= num_to_avg
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = [[], []]
            images_list = []
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
        k += 1
    return images_list, intensity_list


# ---------- fits and plots ---------- #


def print_fit_params(fit_params):
    for key, value in fit_params.items():
        print(f"{key}: {round(value, 2)}")


def plot_fit(params, fit_func=f.positive_cos):
    xdata = np.linspace(0, 255, 256)
    ydata = fit_func(xdata, *params.values())
    return xdata, ydata



# ------------ parameters stuff ------------- #

def default_params():
    params = {}
    params["correspond_to_2pi"] = [256]
    params["subdomain_size"] = [32]
    params["reference_position"] = [(15, 11)]
    params["subdomain_position"] = [(14, 11)]
    params["decline"] = [(1, 1)]
    params["precision"] = [8]
    params["num_to_avg"] = [1]
    return params


def get_params(params):
    print("to retain current value, just press enter")
    params["correspond_to_2pi"].append(get_correspond_to_2pi(last_nonempty(params["correspond_to_2pi"])))
    params["subdomain_size"].append(get_subdomain_size(last_nonempty(params["subdomain_size"])))
    subdomain_size = last_nonempty(params["subdomain_size"])
    params["reference_position"].append(get_position(last_nonempty(params["reference_position"]), subdomain_size, "reference"))
    params["subdomain_position"].append(get_position(last_nonempty(params["subdomain_position"]), subdomain_size, "second"))
    params["decline"].append(get_decline(last_nonempty(params["decline"])))
    params["precision"].append(get_precision(last_nonempty(params["precision"])))
    params["num_to_avg"].append(get_num_to_avg(last_nonempty(params["num_to_avg"])))


def last_nonempty(lst):
    for i in range(1, len(lst) + 1):
        if lst[-i]:
            return lst[-i]
        

def get_correspond_to_2pi(current):
    correspond_to_2pi_to_be = input(f"enter value of pixel corresponding to 2pi phase shift. current value: {current} >> ")
    if correspond_to_2pi_to_be == '':
        return correspond_to_2pi_to_be
    return int(correspond_to_2pi_to_be)


def get_precision(current):
    precision_to_be = input(f"enter number of phase shifts. current value: {current} >> ")
    if precision_to_be == '':
        return precision_to_be
    return int(precision_to_be)

def get_num_to_avg(current):
    num_to_avg_to_be = input(f"enter number of frames to be averaged. current value: {current} >> ")
    if num_to_avg_to_be == '':
        return num_to_avg_to_be
    return int(num_to_avg_to_be)


def get_decline(current):
    while True:
        decline_to_be = input(f"enter decline angle as a tuple in units of quarter of first diffraction maximum. current value: {current} >> ")
        if decline_to_be == '':
            return decline_to_be
        decline_to_be = eval(decline_to_be)
        x_angle, y_angle = decline_to_be
        if x_angle > 4 or y_angle > 4:
            print("neither of angles should exceed 4")
            continue
        return decline_to_be
        

def get_position(current, subdomain_size, which):
    max_height, max_width = c.slm_height // subdomain_size, c.slm_width // subdomain_size
    limits = (max_width, max_height)
    while True:
        position_to_be = input(f"enter position of the {which} subdomain as a tuple of ints.  width: {max_width} subdomains, height: {max_height} subdomains. current value: {current} >> ")
        if position_to_be == '':
            if not check_coord_limits(current, limits):
                continue
            return position_to_be
        position_to_be = eval(position_to_be)
        if not check_coord_limits(position_to_be, limits):
            continue
        x, y = position_to_be
        array_coords = (x, y)
        return array_coords
    

def check_coord_limits(coords, limits):
    x, y = coords
    xmax, ymax = limits
    if x > xmax:
        print(f"first coordinate should not exceed {xmax}")
        return False
    if y > ymax:
        print(f"second coordinate should not exceed {ymax}")
        return False
    return True

def real_subdomain_position(subdomain_position, subdomain_size):
    subdomain_position_x, subdomain_position_y = subdomain_position
    return subdomain_size * subdomain_position_x, subdomain_size * subdomain_position_y


def get_subdomain_size(current):
    while True:
        size_to_be = input(f"enter subdomain size in pixels. current value: {current} >> ")
        if size_to_be == '':
            return size_to_be
        size_to_be = int(size_to_be)
        if size_to_be > c.slm_height:
            print(f"subdomain size should not exceed {c.slm_height}")
            continue
        return size_to_be


# ------------ visualizing -------------- #

def format_output(hologram, frame, intensity_data, intensity_fit, internal_screen_resolution):

    pad = 10

    hologram, frame, crop_coords = resize_hologram_crop_frame(hologram, frame, internal_screen_resolution, pad) # TODO: why do i need to overwrite them?
    plot_image = create_plot_img(intensity_data, intensity_fit, internal_screen_resolution, hologram.height, pad)

    display_image, frame_coords = paste_together(hologram, frame, plot_image, internal_screen_resolution, pad)
    return display_image, (crop_coords, frame_coords)


def paste_together(hologram, frame, plot_image, internal_screen_resolution, pad):
    blank = im.new("L", internal_screen_resolution, 50).convert("RGB")
    blank.paste(hologram, (pad, pad))
    frame_coords = 2 * pad + hologram.width, pad
    blank.paste(frame, (frame_coords))
    blank.paste(plot_image, (pad, 2 * pad + hologram.height))
    return blank, frame_coords



def resize_images(hologram, frame, screen_resolution, pad):
    '''resize two images (hologram & frame) in a way that
    they have the same height and fit exactly in the screen next to each other also with padding
    '''
    min_plot_height = 200
    screen_width, screen_height = screen_resolution
    resize = (screen_width - 3 * pad) / (hologram.width + hologram.height / frame.height * frame.width)
    rest = screen_height - 2 * pad - min_plot_height
    if resize * hologram.height > rest :
        resize = rest / hologram.height
    hologram = hologram.resize((int(resize * hologram.width), int(resize * hologram.height)), im.LANCZOS)
    frame_ratio = hologram.height / frame.height
    frame = frame.resize((int(frame_ratio * frame.width), int(frame_ratio * frame.height)), im.LANCZOS)
    return hologram, frame


def resize_hologram_crop_frame(hologram, frame, screen_resolution, pad):
    '''similar to resize_images except the second one is cropped instead of resized
    so that pixels from the original image correspond to pixels on the screen
    '''
    min_plot_height = 200
    screen_width, screen_height = screen_resolution
    resize = (screen_width - 3 * pad) / (hologram.width + hologram.height / frame.height * frame.width)
    rest = screen_height - 2 * pad - min_plot_height
    if resize * hologram.height > rest :
        resize = rest / hologram.height
    hologram = hologram.resize((int(resize * hologram.width), int(resize * hologram.height)), im.LANCZOS)
    frame_width = screen_width - 3 * pad - hologram.width
    frame, crop_coords = crop_frame(frame, frame_width, hologram.height)    
    return hologram, frame, crop_coords

def crop_frame(frame, width, height):
    original_width, original_height = frame.size
    # middle_coords = (original_width // 2, original_height // 2)
    x_corner = (original_width - width) // 2
    y_corner = (original_height - height) // 2
    crop_coords = (x_corner, y_corner, x_corner + width, y_corner + height)
    cropped_frame = frame.crop(crop_coords)
    return cropped_frame, crop_coords


def create_plot_img(intensity_data, intensity_fit, screen_resolution, hologram_height, pad):
    screen_width, screen_height = screen_resolution
    plot_width = screen_width - 2 * pad
    plot_height = screen_height - hologram_height - 3 * pad
    # Create a subplot for the plot with the specified width and height
    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100)) # i use f-ing magical constants because f-ing matplotlib wants the number in fucking inches
    ax.scatter(intensity_data[0], intensity_data[1], )
    ax.plot(intensity_fit[0], intensity_fit[1])
    ax.set_ylim(0, 256)
    ax.set_xlabel('phase shift')
    ax.set_ylabel('intensity')
    # ax.set_title('')

    # Create a Tkinter canvas for the plot
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Convert the plot to an Image object
    plot_image = im.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    return plot_image


def get_internal_screen_resolution():
    monitors = get_monitors()
    for monitor in monitors:
        # Check if the monitor is on the left side of the screen
        if monitor.x == 0:
            return monitor.width, monitor.height
        


# --------- video creating section ---------- #

def dot_frames(frames, coords):
    frame_img_list = []
    for frame in frames:
        frame_img = im.fromarray(frame, "L").convert("RGB")
        frame_img = add_cross(frame_img, coords)
        # frame_img.show()
        frame_img_list.append(frame_img)
    return frame_img_list

def add_cross(img, coords):
    y, x = coords
    radius = 5
    red = (255, 0, 0)
    for i in range(- radius, radius + 1):
        img.putpixel((x + i, y), red)
    for i in range(- radius, radius + 1):
        img.putpixel((x, y + i), red)
    return img


def make_imgs_for_video(seed, frame_img_list, video_frame_info):
    crop_coords, frame_coords = video_frame_info
    img_list = []
    for frame in frame_img_list:
        cropped_frame = frame.crop(crop_coords)
        new_img = deepcopy(seed)
        new_img.paste(cropped_frame, frame_coords)
        img_list.append(new_img)
    return img_list


def make_name(params):
    name = f""
    for key in params.keys():
        name += f"{key}={last_nonempty(params[key])}_"
    return name + time.strftime("%Y-%m-%d_%H-%M-%S")


def images_to_video(image_list, video_name, fps, output_path="."):
    # Get dimensions from the first image
    width, height = image_list[0].size

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs too, like 'XVID'
    video = cv2.VideoWriter(f"{output_path}/{video_name}.mp4", fourcc, fps, (width, height))

    for img in image_list:
        # Convert PIL Image to numpy array
        frame = np.array(img)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script for simulating and visualizing wavefront_correction loops")
    parser.add_argument('-m', '--mode', choices=['i', 'v'], default='i', type=str, help="i for images, v for video output")
    args = parser.parse_args()

    explore(args)
