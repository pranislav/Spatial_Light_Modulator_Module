'''creates phase mask for LC-SLM which compensates
aberrations caused both by the modulator and whole optical path.
This mask should be added up with any projected hologram.
For each optical there should be generated its own mask.

Principle:
modulator's screen is divided into square subdomains
for each subdomain we are searching for optimal phase offset
optimal phase offset is found as following:
let us have a reference subdomain which deflects light to particular angle
for each subdomain we do following:
the subdomain deflects light to the exact same angle as the reference one
all the others subdomains are "off"
the hologram on the subdomain is shifted by a constant phase several times
the phase shift which causes best constructive interference with the reference subdomain
is chosen and written into the phase mask
the quality of the interference is decided by
measuring intensity at the end of the optical path with a camera
'''

from calibration_lib import *
import numpy as np
import os
import sys
from pylablib.devices import uc480


reference_hologram_coordinates_ratio = ((1, 2), (1, 2)) # example: ((1, 2), (3, 4)) -- y coordinate will be roughly half of slm height, x coordinate will be roughly three quarters of slm width 


def main(path_to_holograms: str, calibration_name: str):
    subdomain_size = get_subdomain_size(path_to_holograms)
    precision = get_precision(f"{path_to_holograms}/0/0")
    cam = uc480.UC480Camera()
    window = create_tk_window()
    H, W = get_number_of_subdomains(subdomain_size)
    # path_to_reference_hologram = get_path_to_reference_hologram(path_to_holograms)
    reference_coordinates = extract_reference_coordinates(reference_hologram_coordinates_ratio, subdomain_size, (H, W))
    reference_hologram = create_reference_hologram(reference_coordinates)
    set_exposure_wrt_reference_img(cam, window, im.fromarray(reference_hologram))
    phase_step = 256 // precision
    phase_mask = np.zeros((H, W))
    # i_0, j_0 = get_reference_position(path_to_reference_hologram)
    i_0, j_0 = reference_coordinates
    # square = detect_bright_area(np.array(im.open(path_to_reference_hologram).convert("L")))
    coordinates = get_highest_intensity_coordinates_img(cam, window, im.fromarray(reference_hologram))
    for i in range(H):
        print(f"{i}/{H}")
        for j in range(W):
            # if i == i_0 and j == j_0: continue # reference subdomain is off and the coordinates are absolute anyway
            top_intensity = 0
            k = 0
            while k < precision:
                hologram_wo_ref = im.open(f"{path_to_holograms}/{i}/{j}/{k}.png")
                hologram = add_ref(hologram_wo_ref, reference_hologram, reference_coordinates, subdomain_size)
                display_image_on_external_screen_img(window, hologram) # displays hologram on an external dispaly (SLM)
                frame = cam.snap()
                # intensity = get_intensity_integral(frame, square)
                intensity = get_intensity_coordinates(frame, coordinates)
                if intensity > top_intensity:
                    top_intensity = intensity
                    phase_mask[i, j] = k * phase_step
                    if intensity == 255:
                        print("maximal intensity was reached")
                        k = 0
                        cam.set_exposure(cam.get_exposure * 1.1) # 10 % increase of exposure time
                k += 1
    name = f"{os.path.basename(path_to_holograms)}_{calibration_name}"
    create_phase_mask(phase_mask, subdomain_size, name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python calibration.py <hologram_set_name> <calibration_name>")
        sys.exit(1)

    
    # Get command line arguments
    path_to_holograms = sys.argv[1]
    calibration_name = sys.argv[2]
# path_to_holograms = "lc-slm/holograms_for_calibration/size32_precision8_x1_y1"
# calibration_name = "home_trial"
    
    # Call main function with the parameters
    main(path_to_holograms, calibration_name)

