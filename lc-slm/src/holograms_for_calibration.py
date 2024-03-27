'''generates hologram series for calibrating SLM'''

from PIL import Image as im
import numpy as np
from random import randint
import constants as c

x_decline = 1 * c.u
y_decline = 1 * c.u
subdomain_size = 8
reference_position = (randint(c.slm_height // subdomain_size), randint(c.slm_width // subdomain_size)) # TODO: dat tam cosi rozumne

def make_hologram(substrate, subdomain_position, phase_shift):
    i_0, j_0 = subdomain_position
    for i in range(subdomain_size):
        for j in range(subdomain_size):
            pass