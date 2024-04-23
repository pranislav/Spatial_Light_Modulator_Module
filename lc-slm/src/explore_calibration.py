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

# TODO: subdomain_position degeneration (real & input one)

import constants as c
import calibration_lib as cl
import display_holograms as dh
import numpy as np
from PIL import Image as im
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pylablib.devices import uc480
from scipy.optimize import curve_fit


def explore():
    params = default_params()
    # sample_list = make_sample_holograms(sample_list, params)
    window = cl.create_tk_window()
    cam = uc480.UC480Camera()
    internal_screen_resolution = get_internal_screen_resolution()
    while True:
        black_hologram = im.fromarray(np.zeros((c.slm_height, c.slm_width)))
        if params["decline"][-1] or params["precision"][-1]:
            angle = last_nonempty(params["decline"])
            precision = last_nonempty(params["precision"])
            sample_list = make_sample_holograms(angle, precision)
        if params["subdomain_size"][-1] or params["reference_position"][-1]:
            subdomain_size = last_nonempty(params["subdomain_size"])
            reference_position = real_subdomain_position(last_nonempty(params["reference_position"]), subdomain_size)
        reference_hologram = cl.add_subdomain(black_hologram, sample_list[0], reference_position, subdomain_size)
        num_to_avg = last_nonempty(params["num_to_avg"])
        if params["decline"][-1] or params["subdomain_size"][-1]:
            cl.set_exposure_wrt_reference_img(cam, window, (256 / 4 - 20, 256 / 4), reference_hologram, num_to_avg)
        hologram = reference_hologram
        subdomain_position = real_subdomain_position(last_nonempty(params["subdomain_position"]), subdomain_size)
        intensity_coord = cl.get_highest_intensity_coordinates_img(cam, window, reference_hologram, num_to_avg)
        frame, intensity_data = calibration_loop_explore(window, cam, hologram, sample_list, subdomain_position, subdomain_size, precision, intensity_coord, num_to_avg)
        fit_params = fit_intensity(intensity_data)
        print_fit_params(fit_params)
        intensity_fit = plot_fit(fit_params)
        display_results(hologram, im.fromarray(frame), intensity_data, intensity_fit, internal_screen_resolution)

        if input("continue (enter) or quit (anything) >> "): break
        get_params(params)


def calibration_loop_explore(window, cam, hologram, sample, subdomain_position, subdomain_size, precision, coordinates, num_to_avg):
    intensity_list = [[], []]
    k = 0
    while k < precision:
        hologram = cl.add_subdomain(hologram, sample[k], subdomain_position, subdomain_size)
        cl.display_image_on_external_screen_img(window, hologram)
        intensity = 0
        for _ in range(num_to_avg):
            frame = cam.snap()
            intensity += cl.get_intensity_coordinates(frame, coordinates)
        intensity /= num_to_avg
        if intensity == 255:
            print("maximal intensity was reached, adapting...")
            cam.set_exposure(cam.get_exposure() * 0.9) # 10 % decrease of exposure time
            k = 0
            intensity_list = [[], []]
            continue
        phase = k * 256 // precision
        intensity_list[0].append(phase)
        intensity_list[1].append(intensity)
        k += 1
    return frame, intensity_list


# ---------- fits and plots ---------- #

def fit_intensity(intensity_data):
    xdata, ydata = intensity_data
    p0 = [100, 100, 1/256, 0]
    params, _ = curve_fit(general_cos, xdata, ydata, p0=p0)
    return params


def print_fit_params(fit_params):
    amplitude_shift, amplitude, frequency, phase_shift = fit_params
    print(f"amplitude_shift: {round(amplitude_shift, 2)}")
    print(f"amplitude: {round(amplitude, 2)}")
    print(f"wavelength: {round(2*np.pi / frequency, 2)}")
    print(f"phase_shift: {round(phase_shift, 2)}")
    print("")


def general_cos(x, amplitude_shift, amplitude, frequency, phase_shift):
    return amplitude_shift + amplitude * np.cos(frequency * (x - phase_shift))


def plot_fit(params):
    amplitude_shift, amplitude, frequency, phase_shift = params
    xdata = np.linspace(0, 255, 256)
    ydata = general_cos(xdata, amplitude_shift, amplitude, frequency, phase_shift)
    return xdata, ydata


# --------- # ---------- #

# TODO: this one compare with the one in cl, if same functionality, move it there, this should be faster
def make_sample_holograms(angle, precision):
    sample = []
    sample.append(cl.decline(angle, precision))
    for i in range(1, precision):
        offset = i * 256 // precision
        sample.append((sample[0] + offset) % 256)
    return sample


# ------------ parameters stuff ------------- #

def default_params():
    params = {}
    params["subdomain_size"] = [32]
    params["reference_position"] = [(15, 11)]
    params["subdomain_position"] = [(14, 11)]
    params["decline"] = [(1, 1)]
    params["precision"] = [8]
    params["num_to_avg"] = [1]
    return params


def get_params(params):
    print("to retain current value, just press enter")
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

def display_results(hologram, frame, intensity_data, intensity_fit, internal_screen_resolution):

    pad = 10

    hologram, frame = resize_images(hologram, frame, internal_screen_resolution, pad)
    plot_image = create_plot_img(intensity_data, intensity_fit, internal_screen_resolution, hologram.height, pad)

    display_image = paste_together(hologram, frame, plot_image, internal_screen_resolution, pad)
    display_image.show()


def paste_together(hologram, frame, plot_image, internal_screen_resolution, pad):
    blank = im.new("L", internal_screen_resolution, 50)
    blank.paste(hologram, (pad, pad))
    blank.paste(frame, (2 * pad + hologram.width, pad))
    blank.paste(plot_image, (pad, 2 * pad + hologram.height))
    return blank



def resize_images(hologram, frame, screen_resolution, pad):
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


def create_plot_img(intensity_data, intensity_fit, screen_resolution, hologram_height, pad):
    screen_width, screen_height = screen_resolution
    plot_width = screen_width - 2 * pad
    plot_height = screen_height - hologram_height - 3 * pad
    # Create a subplot for the plot with the specified width and height
    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100)) # i use f-ing magical constants because f-ing matplotlib wants the number in fucking inches
    ax.scatter(intensity_data[0], intensity_data[1], )
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


if __name__ == "__main__":
    explore()