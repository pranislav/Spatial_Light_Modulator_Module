# Spatial Light Modulator Module

**A module for work with spatial light modulator - SLM (especially transparent phase-only SLM which was this project developed for)**

## Functionality:
- generate analytical hologram (light declining and focusing)
- generate general hologram (two algorithms - GS and GD)
- transform hologram - mostly shifting of the image in Fourier plane
- display hologram on external screen (slm) without panels and stuff
- create gif (holografic, expected images, gif from process of hologram generation)
- make hologram gif for moving traps based on parametrizations of trap paths
- handling non-uniform incomming intensity
- calibrate optical path containing lc-slm (wave front correction)
- postprocess phase masks for wave front correction

### Wave front correction
The process of wave front correction (Tomas Cizmar approach) is implemented in the wavefront_correction.py. Resulting phase mask can compensate curvature of SLM and also aberrations induced by other optical components. The process at default parameters lasts cca 10 min. For stronger noise reduction (especially induced by time varying effects such as fluctuations of light) you can run the process multiple times and average the phase masks using average_masks.py. If you would like to remove defocus compensation from the mask after the process (it can be done within the process using appropriate argument), you can do so by using remove_defocus.py.

- wavefront_correction.py - 

### Hologram generating
(about algorithms, analytical, )

### Hologram displaying

## Usage
- run scripts on command line >> `python src/script.py [arguments]`
- run scripts from project root to avoid unexpected behavior in terms of file structure
- run with flag `-help` to find out about possible arguments and options
- you might want to set constants in constants.py file to match parameters of your SLM and laser light

## Requirements
- monochromatic, coherent plane wave light
- transparent phase-only SLM
- camera Thorlabs uc480 + operating system Windows on your pc. In use of different camera you might need to redo parts of code with camera. Windows is required because uc480 library from pylablib.devices is dependent on some dll.


