# Spatial Light Modulator Module

**A module for work with spatial light modulator - SLM (especially transparent phase-only SLM which was this project developed for). Covers generating & displaying holograms, wave front correction and creating optical traps.**

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
    - [Wave front correction](#wave-front-correction)
    - [Hologram generating](#hologram-generating)
    - [Hologram displaying](#hologram-displaying)
    - [Holograms concerning traps](#holograms-concerning-traps)
    - [SLM calibration](#slm-calibration)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contact](#contact)

## Introduction
This project arised beside working on my thesis entitled "Applications of Spatial Light Modulators". It contains scripts that helped me to perform related experiments, mainly optical trapping. Core aspect of this project is implementation of wave front correction according to Tomas Cizmar's approach. Originally, there was a second part of the project concerning work with Digital Micromirror Device (DMD). Because of lack of time it was not developed, however there are some sketches stashed in src/archive/dmd.


## Features

### Wave front correction
The process of wave front correction (Tomas Cizmar approach) is implemented in the wavefront_correction.py. Resulting phase mask can compensate curvature of SLM and also aberrations induced by other optical components. Brief explanation of how the process works follows. For more detailed information check the [original paper](https://www.nature.com/articles/nphoton.2010.85).

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

- *wavefront_correction.py* - this launches the wave front correction process which produces a phase mask (in fact two phase masks, one raw and one smoothed). The process with default parameters lasts cca 10 min. Explanation of arguments which might be confusing: -nsp, --sqrted_number_of_source_pixels is related to the number of pixels which we read the intensity from. The pixels are chosen in a square area, so this number represents length of the area in number of pixels. Case of nsp=1 corresponds exactly to the process described above, nsp=N corresponds to creation of N**2 phase masks which are averaged at the end of the process (this helps to reduce noise). Parameter -parallel causes that capturing of photos and computation of optimal phase offset run in parallel. This significantly affects the lenght of the process just in case of choosing the optimal phase by fitting & many source pixels, otherwise it does not help much.

- *explore_wavefront_correction.py* - you can run this script to make a picture about what is going on in wave front correction process or to find out what is wrong when the process exhibits unexpected behavior. Use mode -m i for static image or -m v for "video" output (the video is saved in images/explore).

- *average_masks.py* - averages masks stated as arguments. This also helps to reduce noise. Argument -subtract subtracts phase mask given as parameter to this argument.

- *remove_defocus.py* - fits and subtracts quadratic phase (which is connected to defocus compensation) from mask stated in the argument. This can be done also within the wave front correction process using argument -rd, --remove_defocus.

- *fit_maps* - similar to wavefront_correction but beside phase mask (phase shift map) returns maps of all fitted parameters. Point of this script was to point out correlations of fitted parameters as there is a lot of them, especially if the *correspond_to2pi* factor is not established yet while this procedure is used to measure it. There is not much use of it now.


### Hologram generating
Holograms can be generated using one of two implemented iterative algorithms or, in special cases, can be computed analytically. These cases include declining and focusing of incident light. The iterative algorithms which are able to produce general hologram are Gerchberg-Saxton algorithm (GS) and an algorithm using gradient descent (GD). The latter one was inspierd by neural networks and developed by my humble person (not claiming i was the first one). 

- *generate_and_transform_hologram.py* - as an input it takes an image that should be result of hologram displaying. Produces phase-only hologram. The image should appear in the Fourier plane. The input image will be resized to match the dimensions of the SLM; when actually displaying, only square-like pictures will preserve their proportions. Explanation of arguments which might be confusing: -q, --quarterize - the original image is reduced to quarter and pasted to black image of its original size before generating hologram to make sure there will be no overlaps between various aliases of displayed image. -gif - an option to create a gif capturing evolution of the hologram (--gif_type h) or expected image (-gif_type i). It was useful rather for developing to see inside the process, but it still can be useful for setting optimal parameters intuitively. Besides, when generating holograms for non-dimensionless traps (bigger than one pixel), the result is also quite interesting, i recommend to try value of -unsettle argument more than one.

- *analytical_holograms.py* - generates holograms for declining and focusing

- *fit_incomming_intensity.py* - takes as an input a photo of laser intensity profile at the plane where the holograms are displayed, makes a gaussian fit of the intensity and returns picture of the gaussian function with fitted parameters. It works as the intensity photo smoothing (naturally just for gaussian intensity profiles). This picture can be than provided to an generating algorithm through *generate_and_transform_holograms.py* as `--incomming_intensity` argument. (Default value of this argument results in uniform intensity, which should be sufficient approximation.)


### Hologram displaying

- *display_holograms.py* - displays holograms on given paths without window porter or task bar on SLM. Allows displaying with mask (wich can be changed any time) and displaying of hologram sequence, which can be stopped or navigated frame-by frame. Another feature is mode for instant wave-front-correction-holograms displaying, which generates and displays holograms used in wave front correction procedure corresponding to entered parameters.


### Holograms concerning traps
Scripts for work with optical traps.

- *multi_decline_generate* - generates static holograms of multiple traps. There can be set shape and size of traps, which does not make sense, but results in holograms of nontrivial aesthetic value.

- *generate_traps_images* - generates series of images with traps from given parametrization of their movement. This serves as an input for *generate_hologram_sequence* which creates holograms corresponding to the traps images.

- *generate_hologram_sequence* - generates holograms coresponding to given images of traps (or any images which are in one directory and their names are unique numbers from zero to number of the images - 1).


### SLM calibration
To set the correct phase shift on given pixel of SLM, you need to know the relation between the color (8-bit integer) on hologram pixel and induced phase shift on SLM pixel. It depends on used laser light and in general the relation can be nonlinear. In this project we work with assumption the relationship is linear, co the outcome of the calibration is one number, "correspond_to_2pi", i.e. 8-bit integer corresponding to two pi phase change. This is the weak part of the project as it does not give consistent results. I recommend to try another approach.

- *color_phase_relation* - script for measuring relation between color of pixel on hologram and corresponding real phase shift on SLM. Linear relation is presumed, so the outcome is *correspond_to_2pi* constant. It is based on wavefront correction procedure with fitting all parameters (period is not fixed as it is measured).

- *color_phase_relation_two_big* - same purpose and process as in *color_phase_relation* except just two big subdomains are used and the process is repeated many times. Big subdomains are used for better interference pattern.


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
