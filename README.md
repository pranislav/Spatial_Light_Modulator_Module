# Spatial Light Modulator Module

**A module for work with spatial light modulator - SLM (especially transparent phase-only SLM which was this project developed for). Covers generating & displaying holograms, wave front correction and related stuff.**

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
    - [Wave front correction](#wave-front-correction)
    - [Hologram generating](#hologram-generating)
    - [Hologram displaying](#hologram-displaying)
    - [Holograms concerning traps](#holograms-concerning_traps)
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


### Hologram generating
Holograms can be generated using one of two implemented iterative algorithms or, in special cases, can be computed analytically. These cases include declining and focusing of incident light. The iterative algorithms which are able to produce general hologram are Gerchberg-Saxton algorithm (GS) and an algorithm using gradient descent (GD). The latter one was inspierd by neural networks and developed by my humble person (not claiming i was the first one). 

- *generate_and_transform_hologram.py* - as an input it takes an image that should be result of hologram displaying. Produces phase-only hologram. The image should appear in the Fourier plane. The input image will be resized to match the dimensions of the SLM; when actually displaying, only square-like pictures will preserve heir proportions. Explanation of arguments which might be confusing: -q, --quarterize - the original image is reduced to quarter and pasted to black image of its original size before generating hologram to make sure there will be no overlaps between various aliases of displayed image. -gif - an option to create a gif capturing evolution of the hologram (--gif_type h) or expected image (-gif_type i). It was useful rather for developing to see inside the process, but it still can be useful for setting optimal parameters intuitively. Besides, when generating holograms for non-dimensionless traps (bigger than one pixel), the result is also quite interesting, i recommend to try value of -unsettle argument more than one.

- *analytical_holograms.py* - generates holograms for declining and focusing

- 

<!-- (about algorithms, analytical,  - handling non-uniform incomming intensity,  ) -->
<!-- - create gif (holografic, expected images, gif from process of hologram generation) -->
<!-- - transform hologram - mostly shifting of the image in Fourier plane -->


### Hologram displaying
To make sure that a hologram is displayed properly, without window porter or task bar, you can use a 


### Holograms concerning traps
<!-- - make hologram gif for moving traps based on parametrizations of trap paths -->



### SLM calibration
To set the correct phase shift on given pixel of SLM, you need to know the relation between the color (8-bit integer) on hologram pixel and induced phase shift on SLM pixel. It depends on used laser light and in general the relation can be nonlinear. In this project we work with assumption the relationship is linear, co the outcome of the calibration is one number, "correspond_to_2pi", i.e. 8-bit integer corresponding to two pi phase change. This is the weak part of the project as it does not give consistent results. I recommend to try another approach. 


## Requirements
- monochromatic, coherent plane wave light
- transparent phase-only SLM
- camera Thorlabs uc480 + operating system Windows on your pc. In use of different camera you might need to redo parts of code with camera. Windows is required because uc480 library from pylablib.devices is dependent on some dll.


## Installation
To install the project, follow these steps:

1. If you want to make changes to the project and keep them version-controlled, please [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository to your GitHub account before cloning. 

2. Clone the repository (forked or original) to your local machine:
   `git clone https://github.com/pranislav/Spatial_Light_Modulator_Module.git`

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
