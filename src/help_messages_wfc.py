'''Help messages for files familiar with wavefront correction procedure
'''


samples_per_period = "number of intensity measurements per one subdomain \
    and also number of subdomain hologram phase shifts"
subdomain_size = "size of subdomain side in pixels"
deflect = "angle to deflect the light in x and y direction (in constants.u unit)"
ct2pi = "value of pixel corresponding to 2pi phase shift"
reference_subdomain_coordinates = "subdomain-scale coordinates of the reference subdomain. \
    Default parameter assigns the reference subdomain to the middle one."
skip_subdomains_out_of_inscribed_circle = "subdomains out of the inscribed circle will not be callibrated. \
    use when the SLM is not fully illuminated and the light beam is circular."
shuffle = "subdomains will be calibrated in random order"
intensity_coordinates = "coordinates of the point where the intensity is measured. \
    if not provided, the point will be found automatically."
choose_phase = "method of finding the optimal phase shift"
sqrted_number_of_source_pixels = "length (in pixels) of side of a square area \
    on photo from which the intensity is taken"
remove_defocus_compensation = "remove defocus compensation from the phase mask"
