import numpy as np
from scipy.optimize import curve_fit
from PIL import Image as im
import constants as c

# Generate some example data
data = "paper_shade_01"
Z = np.array(im.open(f"lc-slm/images/incomming_intensity_images/{data}.png").convert("L")) / 256
y_len, x_len = Z.shape
x_data = np.linspace(0, x_len, x_len)
y_data = np.linspace(0, y_len, y_len)
X, Y = np.meshgrid(x_data, y_data)
R_cam = x_len / 20 * 7


# Define the function to fit
def gaussian_step(xy, a, b, x0, y0):
    # x0 = x_len // 2
    # y0 = y_len // 2
    x, y = xy
    return b + np.exp(-a * ((x - x0)**2 + (y - y0)**2)) * heaviside_circle(x, y, x0, y0, R_cam)


def heaviside_circle(x, y, x0, y0, R):
    len = x.shape[0]
    z = np.zeros(len)
    for i in range(len):
        if (x[i] - x0) ** 2 + (y[i] - y0) ** 2 < R ** 2:
            z[i] = 1
    return z


# Flatten the data for curve_fit
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()

# Perform the fitting
# print("R approximately", x_len / 20 * 7)
p0 = [2e-05, 0.05, x_len // 2, y_len // 2]
popt, pcov = curve_fit(gaussian_step, (x_flat, y_flat), z_flat, p0=p0)

# Output the fitted parameters
a_fit, b_fit, x0_fit, y0_fit = popt
print("Fitted parameters:")
print("a =", a_fit)
print("b = ", b_fit)
print("x0 =", x0_fit)
print("y0 =", y0_fit)
# print("R = ", R_fit)
# print(pcov)

gaustep = (256 * (b_fit + gaussian_step((x_flat, y_flat), a_fit, b_fit, x0_fit, y0_fit))).reshape(Z.shape)
# im.fromarray(gaustep).show()


def gaussian_step_coord(x, y, a, x0, y0, R):
    return (np.exp(-a * ((x - x0)**2 + (y - y0)**2))) * heaviside_circle_coord(x, y, x0, y0, R)

def heaviside_circle_coord(x, y, x0, y0, R):
    if (x - x0) ** 2 + (y - y0) ** 2 < R ** 2:
        return 1
    return 0


def create_intensity_mask(a):
    intensity_mask_arr = np.zeros((c.slm_height, c.slm_width))
    x0 = c.slm_width / 2
    y0 = c.slm_height / 2
    R_slm = c.slm_height / 2
    convert = R_cam / R_slm
    for y in range(intensity_mask_arr.shape[0]):
        y_ = y * convert
        for x in range(intensity_mask_arr.shape[1]):
            x_ = x * convert
            intensity_mask_arr[y, x] = 255 * gaussian_step_coord(x_, y_, a, x0*convert, y0*convert, R_slm*convert)
    im.fromarray(intensity_mask_arr).convert("L").save(f"lc-slm/incomming_intensity_images/{data}_b=0_intensity_mask.png")

create_intensity_mask(a_fit)