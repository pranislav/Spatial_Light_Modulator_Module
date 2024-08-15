"""this thing helps me to optimize optical path wrt. laser power use
and Cmos chip use with restriction to available lenses
works just for given optical path"""

# optical path:
# source -> collimator          ------> SLM           ------> telescope_in --------> telescope_out -----> microscope   --------- or camera_lens -> CCD_chip
#           f_col, diameter_col width1  small_dim_SLM width2  f_in         distance1 f_out         width3 diameter_mic distance2    f_cam          small_dim_CCD
#                                       large_dim_SLM                                                                                              large_dim_CCD

import constants as c

# fixed stuff in mm
small_dim_SLM = 28
large_dim_SLM = 38
f_mic = 9
small_dim_CCD = 5
large_dim_CCD = 6
diameter_mic = 9


# realtions
def distance1(f_in):
    return f_in * c.first_diff_max


def distance2(f_in, f_out):
    return distance1(f_in) * f_mic / f_out


def f_cam(f_in, f_out):
    return ("less_than", small_dim_CCD / distance1(f_in) * f_out)


def telescope_magnification(f1, f2):
    return f2 / f1


def width3(width1, f_in, f_out):
    width2 = min(width1, large_dim_SLM)
    return telescope_magnification(f_in, f_out) * width2


def is_efficient_mic(width1, f_in, f_out):
    return width3(width1, f_in, f_out) < diameter_mic


def is_efficient_SLM(width1):
    return width1 < small_dim_SLM
