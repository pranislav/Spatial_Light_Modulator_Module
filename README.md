# Spatial Light Modulator Module

**A module for work with transparent phase-only spatial light modulator (SLM). Contains scripts for generating & displaying holograms, wave front correction and creating optical traps.**

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
    - [Wave front correction](#wave-front-correction)
    - [Hologram generating](#hologram-generating)
    - [Hologram displaying](#hologram-displaying)
    - [Optical trapping tools](#optical-trapping-tools)
    - [Greyscale - phase response determining](#greyscale---phase-response)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contact](#contact)

## Introduction
This project arised beside working on my thesis entitled [Applications of Spatial Light Modulators](https://is.muni.cz/auth/th/erkjd/Diplomka-final.pdf). It contains scripts that helped me to perform related experiments, mainly optical trapping. Core aspect of this project is implementation of wave front correction according to Tomas Cizmar's approach. Originally, there was a second part of the project concerning work with Digital Micromirror Device (DMD). Because of lack of time it was not developed, however there are some sketches stashed in src/archive/dmd.


## Features

### Wave front correction
The process of wave front correction (Tomas Cizmar approach) is implemented in the *wavefront_correction.py*. Resulting phase mask can compensate curvature of SLM and also aberrations induced by other optical components. Brief explanation of how the process works follows. For more detailed information check the [original paper](https://www.nature.com/articles/nphoton.2010.85) or [my thesis](https://is.muni.cz/auth/th/erkjd/Diplomka-final.pdf) (in slovak).

Modulator's screen is divided into square subdomains (groups of pixels) of equal size. 
For each subdomain we are searching for optimal phase offset.
Once the phase offset is found, it is recorded to corresponding element of the phase mask.
At the beginning of the process we choose a reference subdomain which deflects light to particular angle all the time.
Then we iterate through rest of the subdomains.
Process of calibrating one subdomain is this:
We let the subdomain deflect the light to the same angle as the reference one while all the other subdomains are "off".
This causes interference fringes which are captured by a camera.
The hologram on the subdomain is shifted by a constant phase several times, causing another interference patterns which are recorded again.
On the captured photos we fix one pixel from which we read intensity of the light.
The coordinates of the pixel are the same for all the subdomains.
Based on information of performed phase shifts and corresponding intensities we choose a phase shift which would correspond to maximal intensity.
This is the phase offset which is recorded on the phase mask.

- *wavefront_correction.py* - this launches the wave front correction process which produces a phase mask in three instances (image (.png) for preview, numpy array (.npy) (with keyword 'smoothed' in name) for use and another numpy array for further processing).
The process with default parameters lasts cca 10 min.
(The argument -nsp, --sqrted_number_of_source_pixels is related to the number of pixels which the intensity is read from.
The pixels are chosen in a square area, so this number represents length of the area in number of pixels.
Case of nsp=1 corresponds exactly to the process described above, nsp=N corresponds to creation of N**2 phase masks which are averaged at the end of the process (this helps to reduce noise).)


- *explore_wavefront_correction.py* - you can run this script to make a picture about what is going on in wave front correction process or to find out what is wrong if the process exhibits unexpected behavior.
Use mode -m i for static image or -m v for "video" output.

- *average_or_subtract_masks.py* - averages masks stated as arguments.
This also helps to reduce noise.
Argument -subtract subtracts phase mask given as parameter to this argument.

- *remove_defocus.py* - fits and subtracts quadratic phase (which is related to defocus compensation) from mask stated in the argument.
This can be done also within the wave front correction process using argument -rd, --remove_defocus.

- *fit_maps* - similar to wavefront_correction but beside phase mask (phase shift map) returns maps of all fitted parameters.
Point of this script was to point out correlations of fitted parameters as there is a lot of them, especially if the *correspond_to2pi* factor is not established yet while this procedure is used to measure it.
There is not much use of it now.


### Hologram generating
Holograms can be generated using one of two implemented iterative algorithms or, in special cases, can be computed analytically.
These cases include deflecting and focusing of incident light.
The iterative algorithms which are able to produce general hologram are Gerchberg-Saxton algorithm (GS) and an algorithm based on gradient descent (GD).
The latter one was inspierd by neural networks and developed by my humble person (not claiming i was the first one). 

- *generate_hologram.py* - as an input it takes an image that should be result of hologram displaying.
Produces phase-only hologram saved as numpy array.
To visualize the hologram use *show_hologram.py*.
Demanded image should appear in the Fourier plane (wrt. SLM).
The program can be used for generating deflect/lens hologram if the input image is not provided and values of `-decline` or `-lens` are set.
The input image is padded with black pixels to make it square so proportions of the reconstructed image are preserved.
<!-- `-gif` - an option to create a gif capturing evolution of the hologram (`--gif_type h`) or expected image (`-gif_type i`).
It was useful rather for developement to see inside the process, but it still can be useful for setting optimal parameters intuitively.
Besides, when generating holograms for non-dimensionless traps (bigger than one pixel), the hologram gif is also quite interesting, i recommend to try value of -unsettle argument more than one. -->



### Hologram displaying

- *display_holograms.py* - displays holograms on given paths without window porter or task bar on SLM.
Allows displaying with mask (wich can be changed any time) and displaying of hologram sequence, which can be stopped or navigated frame-by frame.
Another feature is mode for instant wave-front-correction holograms displaying, which generates and displays holograms used in wave front correction procedure corresponding to entered parameters.


### Optical trapping tools
Scripts for work with optical traps.

- *move_traps.py* - enables to controll position of an optical trap (or two traps) with keyboard.
Requires SLM as part of an optical tweezers placed in a way that the traps are on Fourier plane wtr. it.

- *generate_traps_image_sequence.py* - generates series of images with traps from given parametrization of their movement.
This serves as an input for *generate_hologram_sequence* which creates holograms corresponding to the traps images.

- *generate_hologram_sequence.py* - generates holograms coresponding to given images of traps (or any images which are in one directory and their names are unique numbers from zero to number of the images - 1).


### Greyscale - phase response
To set the correct phase shift on given pixel of SLM, you need to know the relation between the greyscale value (8-bit integer) on hologram pixel and induced phase shift on SLM pixel.
It depends on used laser light and in general the relation can be nonlinear.
In this project we work with assumption the relationship is linear, so the outcome of the calibration is one number, "correspond_to_2pi", i.e. 8-bit integer corresponding to two pi phase change.
This is the weak part of the project as it does not give consistent results.
I recommend to try another approach.

- *color_phase_relation.py* - script for measuring relation between color (greyscale value) of pixel on hologram and corresponding real phase shift on SLM.
Linear relation is presumed, so the outcome is *correspond_to_2pi* constant.
It is based on wavefront correction procedure with fitting all parameters (period is not fixed as it is measured).

- *color_phase_relation_two_big.py* - same purpose and process as in *color_phase_relation* except just two subdomains are used and the process is repeated many times.
Big subdomains are recommended for better interference pattern.
<!-- TODO: na velkosti nezide, ta sa da nastavit aj v tom druhom. Spomenut, ze sa da druhy krat fitovat s fixnutymi parametrami -->


## Requirements
- monochromatic, coherent plane wave light
- transparent phase-only SLM
- camera Thorlabs uc480 + operating system Windows on your pc. In use of different camera you might need to redo parts of code with camera. Windows is required because uc480 library from pylablib.devices is dependent on some dll.
- other equipement in case of optical trapping


## Installation
To install the project, follow these steps:

1. If you want to make changes to the project and keep them version-controlled, please [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository to your GitHub account before cloning. 

2. Clone the repository to your local machine:

   `git clone https://github.com/pranislav/Spatial_Light_Modulator_Module.git`

   For clonning forked project replace 'pranislav' with your account name.

3. Navigate to the project directory:
    `cd Spatial_Light_Modulator_Module`

4. Install dependencies using [pipenv](https://pipenv.pypa.io/en/latest/):
    `pipenv install`

5. Activate the virtual environment:
    `pipenv shell`


## Usage
- run scripts on command line `>> python src/script.py [arguments]`
- run scripts from project root to avoid unexpected behavior in terms of file structure
- run with flag `-h` or `--help` to see available options and usage instructions
- you might want to set constants in constants.py file to match parameters of your SLM and laser light

## Contact
You can reach me via email: brankopaluch@gmail.com
