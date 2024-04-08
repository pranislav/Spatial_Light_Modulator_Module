import numpy as np
from scipy.optimize import curve_fit
from PIL import Image as im

# Generate some example data
Z = np.array(im.open("lc-slm/incomming_intensity_images/paper_shade_01.png").convert("L")) / 256
y_len, x_len = Z.shape
x_data = np.linspace(0, x_len, x_len)
y_data = np.linspace(0, y_len, y_len)
X, Y = np.meshgrid(x_data, y_data)

# Define the function to fit
def gaussian_step(xy, a, b, x0, y0):
    R = x_len / 20 * 7
    # x0 = x_len // 2
    # y0 = y_len // 2
    x, y = xy
    return b + np.exp(-a * ((x - x0)**2 + (y - y0)**2)) * heaviside_circle(x, y, x0, y0, R)


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
p0 = [1.6e-06, 0, x_len // 2, y_len // 2]
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
im.fromarray(gaustep).show()
