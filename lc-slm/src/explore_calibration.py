'''script which interacts with the experiment and provides
an insight into the experiment response to variaous parameters change
it displays a calibration hologram specified by user on the SLM
and then shows the hologram, an image from camera and
evolution of intensity with respect to phase shift of current subdomain
fitted with cosine and prints out fitted parameters

main goal is to help the user to estimate optimal configuration for
calibration and color-phase relation search

parameters to be changed:
- subdomain size
- position of subdomain
- position of reference subdomain
- declining angle
- nuber of phase shifts
- number of frames to average to supress the fluctuation
'''

import constants as c
import calibration_lib as cl
import display_holograms as dh
import numpy as np
from PIL import Image as im, ImageTk
from screeninfo import get_monitors
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pylablib.devices import uc480
from scipy.optimize import curve_fit


def explore():
    params = default_params()
    sample_list = make_sample_holograms(sample_list, params)
    window = cl.create_tk_window()
    cam = uc480.UC480Camera()
    while not input("continue (enter) or quit (anything) >> "):
        black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
        if params["decline"][-1] or params["precision"][-1]:
            angle = last_nonempty(params["decline"])
            precision = last_nonempty(params["precision"])
            sample_list = make_sample_holograms(angle, precision)
        if params["subdomain_size"][-1] or params["reference_position"][-1]:
            subdomain_size = last_nonempty(params["subdomain_size"])
            reference_position = last_nonempty(params["reference_position"])
        reference_hologram = cl.add_subdomain(black_hologram, sample_list[0], reference_position, subdomain_size)
        if params["decline"] or params["subdomain_size"]:
            cl.set_exposure_wrt_reference_img((256 / 4 - 20, 256 / 4), cam, window, reference_hologram)
        hologram = reference_hologram
        subdomain_position = last_nonempty(params["subdomain_size"])
        num_to_avg = last_nonempty(params["num_to_avg"])
        intensity_coord = cl.get_highest_intensity_coordinates_img(cam, window, reference_hologram)
        frame, intensity_data = calibration_loop_explore(window, cam, hologram, sample_list, subdomain_position, subdomain_size, precision, intensity_coord, num_to_avg)
        intensity_fit = plot_fit(fit_intensity(intensity_data))
        display_results(hologram, frame, intensity_data, intensity_fit)

        # if input("continue (enter) or quit (anything) >> "): break
        get_params(params)


def calibration_loop_explore(window, cam, hologram, sample, subdomain_position, subdomain_size, precision, coordinates, num_to_avg):
    intensity_list = [[], []]
    k = 0
    while k < precision:
        hologram = cl.add_subdomain(hologram, sample[k], subdomain_position, subdomain_size)
        cl.display_image_on_external_screen_img(window, hologram)
        for _ in range(num_to_avg):
            frame = cam.snap()
            intensity += cl.get_intensity_coordinates(frame, coordinates)
        intensity /= num_to_avg
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
        k += 1
    return frame, intensity_list


# ---------- fits and plots ---------- #

def fit_intensity(intensity_data):
    xdata, ydata = intensity_data
    params, _ = curve_fit(general_cos, xdata, ydata)
    return params


def general_cos(x, amplitude_shift, amplitude, frequency, phase_shift):
    return amplitude_shift + amplitude * np.cos(frequency * x - phase_shift)


def plot_fit(params):
    amplitude_shift, amplitude, frequency, phase_shift = params
    xdata = np.linspace(0, 255, 256)
    ydata = general_cos(xdata, amplitude_shift, amplitude, frequency, phase_shift)
    return xdata, ydata


# --------- # ---------- #

# TODO: this one compare with the one in cl, if same functionality, move it there, this should be faster
def make_sample_holograms(angle, precision):
    sample = []
    sample[0] = cl.decline(angle, precision)
    for i in range(1, precision):
        offset = i * 256 // precision
        sample[i] = (sample[0] + offset) % 256


# ------------ parameters stuff ------------- #

def default_params():
    params = {}
    params["subdomain_size"] = [32]
    params["reference_position"] = [(12, 16)]
    params["subdomain_position"] = [(12, 15)]
    params["decline"] = [(1, 1)]
    params["precision"] = [8]
    params["num_to_avg"] = [1]
    return params


def get_params(params):
    print("to retain current value, just press enter")
    params["subdomain_size"].append(get_subdomain_size(last_nonempty(params["subdomain_size"])))
    params["reference_position"].append(get_position(last_nonempty(params["reference_position"]), params["subdomain_size"]))
    params["subdomain_position"].append(get_position(last_nonempty(params["subdomain_position"]), params["subdomain_size"]))
    params["decline"].append(get_decline(last_nonempty(params["decline"])))
    params["precision"].append(get_precision(last_nonempty(params["precision"])))
    params["num_to_avg"].append(get_num_to_avg(last_nonempty(params["num_to_avg"])))


def last_nonempty(lst):
    for i in len(lst):
        if lst[-i]:
            return lst[-i]


def get_precision(current):
    return input(f"enter number of phase shifts. current value: {current} >> ")

def get_num_to_avg(current):
    return input(f"enter number of frames to be averaged. current value: {current} >> ")


def get_decline(current):
    while True:
        decline_to_be = input(f"enter decline angle as a tuple in units of quarter of first diffraction maximum. current value: {current} >> ")
        if decline_to_be == '':
            return decline_to_be
        x_angle, y_angle = decline_to_be
        if x_angle > 4 or y_angle > 4:
            print("neither of angles should exceed 4")
            continue
        return decline_to_be
        

def get_position(current, subdomain_size):
    max_height, max_width = c.slm_height // subdomain_size, c.slm_width // subdomain_size
    while True:
        position_to_be = input(f"enter position of the reference subdomain as a tuple of ints. height: {max_height} subdomains, width: {max_width} subdomains. current value: {current} >> ")
        if position_to_be == '':
            return position_to_be
        x, y = position_to_be
        if x > max_width:
            print(f"first coordinate should not exceed {max_width}")
            continue
        if y > max_height:
            print(f"second coordinate should not exceed {max_height}")
            continue
        return position_to_be

def get_subdomain_size(current):
    while True:
        size_to_be = input(f"enter subdomain size in pixels. current value: {current} >> ")
        if size_to_be == '':
            return size_to_be
        if size_to_be > c.slm_height:
            print(f"subdomain size should not exceed {c.slm_height}")
            continue
        return size_to_be


# ------------ visualizing -------------- #

def display_results(hologram, frame, intensity_data, intensity_fit):

    pad = 10

    # Get screen resolution
    screen_resolution = get_internal_screen_resolution()

    # Create a Tkinter window
    root = tk.Tk()
    # root.title("Images and Plot Display") # TODO: maybe parameters here?

    resize_images(hologram, frame, screen_resolution, pad)

    plot_image = create_plot_img(intensity_data, intensity_fit, screen_resolution, hologram.height, pad)

    # Convert images to Tkinter PhotoImage objects
    tk_hologram = ImageTk.PhotoImage(hologram)
    tk_frame = ImageTk.PhotoImage(frame)
    tk_plot_image = ImageTk.PhotoImage(plot_image)

    # Display images and plot
    plot_label = tk.Label(root, image=tk_plot_image)
    plot_label.image = tk_plot_image
    plot_label.pack(side="top", padx=pad, pady=pad)

    image_label1 = tk.Label(root, image=tk_hologram)
    image_label1.image = tk_hologram
    image_label1.pack(side="left", padx=pad, pady=pad)

    image_label2 = tk.Label(root, image=tk_frame)
    image_label2.image = tk_frame
    image_label2.pack(side="left", padx=pad, pady=pad)

    # Run the Tkinter event loop
    root.update_idletasks()


def resize_images(hologram, frame, screen_resolution, pad):
    screen_width, screen_height = screen_resolution
    resize = screen_width - 3 * pad / (hologram.width + hologram.height / frame.height * frame.width)
    hologram.resize((resize * hologram.width, resize * hologram.height), im.ANTIALIAS)
    frame_ratio = hologram.height / frame.height * resize
    frame.resize((frame_ratio * frame.width, frame_ratio * frame.height), im.ANTIALIAS)


def create_plot_img(intensity_data, intensity_fit, screen_resolution, hologram_height, pad):
    screen_width, screen_height = screen_resolution
    plot_width = screen_width - 2 * pad
    plot_height = screen_height - hologram_height - 3 * pad
    # Create a subplot for the plot with the specified width and height
    fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=100)  # Use dpi to control pixel density
    ax.plot(intensity_data[0], intensity_data[1])
    ax.plot(intensity_fit[0], intensity_fit[1])
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
