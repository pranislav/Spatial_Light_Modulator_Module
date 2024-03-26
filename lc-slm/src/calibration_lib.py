import constants as c

def get_number_of_subdomains(subdomain_size):
    if c.slm_height % subdomain_size != 0 or c.slm_width % subdomain_size != 0:
        print(f"some of the SLM pixels won't be covered,
              you better choose number which divides {c.slm_height} and {c.slm_width}")
    return c.slm_height//subdomain_size, c.slm_width//subdomain_size


def get_intensity(img):
    '''returns intensity of relevant spot'''
    pass


def create_phase_mask(phase_mask, name):
    '''creates and saves phase mask image based on phase mask array'''
    pass
