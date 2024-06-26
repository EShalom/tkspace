[![DOI](https://zenodo.org/badge/665593922.svg)](https://zenodo.org/doi/10.5281/zenodo.10056112)

# tkspace package

This package tkspace houses a number of sub-packages and modules which 
describe spatiotemporal tracer kinetics in one and two compartment systems. 
This includes both simulation of concentration time curve data from physical 
quantities such as influx and velocity and algoritms for the inverse problem 
concerned with finding these quantities from the output data alone.

# Overview

## Installation and Dependencies

Install the tkspace package using the .whl file:  

* Navigate to the dist directory  
* Run `pip install tkspace_pyess-0.0.1-py3-none-any.whl`   
* This will install the tkspace as a package within your current python environment  

Runs using Python3.x, other specifc library dependancies are as follows:  

* numpy (developed with verison 1.20.1)
    - `pip install numpy==1.20.1` 
    - Comes pre-installed with most python builds
* scipy (developed with 1.7.1)
    - `pip install scipy==1.7.1`
    - Comes pre-installed with most python builds
* sigfig from PyPI  
    - `pip install sigfig`
    - For significant figure rounding.
* tailer from PyPI 
    - `pip install tailer`
    - For reading the last N lines of files.

---
## Documentation

[Documentation](https://EShalom.github.io/tkspace/) is under construction.

---

## Example Scripts

See the example scripts in the example directory for setups showing full implementation in 3 two-compartment system cases.
This includes:

* The forward model for producing a ground truth set of concentration curves.
* The addition of sampling and noise to these data.
* The application of an inversion recovery using the gradient descent method.

The functions are imported with general names so they are easy to follow and compare.

---

## Full Implementation 
See a related repository [tkspace examples and outputs](https://github.com/EShalom/tkspace_examples_and_outputs) with [DOI](https://zenodo.org/doi/10.5281/zenodo.10870945) For runfiles and analysis scripts that relate to simulations and results presented in the manuscript (Shalom ES, Van Loo S, Khan A, and Sourbron SP. _Submitted_. "Identifiability of spatiotemporal tissue perfusion models".).

The runfiles within the [tkspace examples and outputs](https://github.com/EShalom/tkspace_examples_and_outputs) require the `tkspace` source code to be built from source using the .whl files provided within the `dist/` directory of this repository.

---

## Structure

The package source files are located in 'src/tkspace'. With the distribution installation file in 'dist'.
The package contains sub-packages with modules to execute:

* Forwards simulation of spatiotemporal tracer kinetic modelling in one and
two compartment systems.
    - `forward_models`
* Algorithms for the inverse problem of recovering tracer kinetic parameters
from the spatio-temporal concnetration data sets.
    - `inversion_methods`

---
