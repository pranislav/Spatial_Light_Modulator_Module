"""this file contains function for fitting cosine in form
amplitude_shift + amplitude * (1/2 + cos(wavelength * x + phase_shift))
with various combinations of fixed parameters
"""

import numpy as np
from scipy.optimize import curve_fit


def fit_intensity_general(intensity_data, func, bounds="N"):
    initial_guess, lower_bounds, upper_bounds = (
        bounds_N() if bounds == "N" else bounds_2pi()
    )
    xdata, ydata = intensity_data
    param_names = func.__code__.co_varnames[1:]
    p0 = [initial_guess[key] for key in param_names]
    lower_bounds = [lower_bounds[key] for key in param_names]
    upper_bounds = [upper_bounds[key] for key in param_names]
    params, _ = curve_fit(
        func, xdata, ydata, p0=p0, bounds=(lower_bounds, upper_bounds)
    )
    params_dict = dict(zip(param_names, params))
    return params_dict


def bounds_N():
    initial_guess = {
        "amplitude_shift": 0,
        "wavelength": 256,
        "phase_shift": 100,
        "amplitude": 256,
    }
    lower_bounds = {
        "amplitude_shift": 0,
        "wavelength": 100,
        "phase_shift": 0,
        "amplitude": 0,
    }
    upper_bounds = {
        "amplitude_shift": 256,
        "wavelength": 300,
        "phase_shift": 256,
        "amplitude": 256,
    }
    return initial_guess, lower_bounds, upper_bounds


def bounds_2pi():
    initial_guess = {
        "amplitude_shift": 0,
        "wavelength": 2 * np.pi,
        "phase_shift": 0,
        "amplitude": 256,
    }
    lower_bounds = {
        "amplitude_shift": 0,
        "wavelength": 0.5,
        "phase_shift": -np.pi,
        "amplitude": 0,
    }
    upper_bounds = {
        "amplitude_shift": 256,
        "wavelength": 3 * np.pi,
        "phase_shift": np.pi,
        "amplitude": 256,
    }
    return initial_guess, lower_bounds, upper_bounds


def positive_cos(x, amplitude_shift, amplitude, wavelength, phase_shift):
    return amplitude_shift + (amplitude / 2) * (
        1 + np.cos((2 * np.pi / wavelength) * (x - phase_shift))
    )


def positive_cos_floor(amplitude_shift=0):
    return lambda x, amplitude, wavelength, phase_shift: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )


def positive_cos_fixed_amp(amplitude):
    return lambda x, amplitude_shift, wavelength, phase_shift: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )


def positive_cos_fixed_wavelength(wavelength):
    return lambda x, amplitude_shift, amplitude, phase_shift: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )


def positive_cos_floor_fixed_amp(amplitude, amplitude_shift=0):
    return lambda x, wavelength, phase_shift: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )


def positive_cos_floor_fixed_wavelength(wavelength, amplitude_shift=0):
    return lambda x, amplitude, phase_shift: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )


def positive_cos_wavelength_only(amplitude_shift, amplitude, phase_shift):
    return lambda x, wavelength: positive_cos(
        x, amplitude_shift, amplitude, wavelength, phase_shift
    )
