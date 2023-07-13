## Overview

This package tkspace  houses a number of sub-packages and modules which 
describe spatiotemporal tracer kinetics in one and two compartment systems. 
This includes both simulation of concentration time curve data from physical 
quantities such as influx and velocity and algoritms for the inverse problem 
concerned with finding these quantities from the output data alone.

---

## Structure

The package contains sub-packages with modules to execute:

* Forwards simulation of spatiotemporal tracer kinetic modelling in one and 
two compartment systems.  
    - `forward_models`
* Algorithms for the inverse problem of recovering tracer kinetic parameters 
from the spatio-temporal concnetration data sets.  
    - `inversion_methods`

---

## Example Scripts

See the example scripts in example directory for a setups showing: 
 
* The forwards model producing a ground truth set of concentration curves.  
* The addition of smapling and noise to these data.  
* The application of an inversion recovery using the gradient descent method.  

There are 4 package examples scripts for each of the 1d forwards model types. 
The functions are imported with general names so they are easy to follow and compare.

---

## Installation and Dependencies

Install the tkspace package using the .whl file:  

* Navigate to the dist directory  
* Run `pip install TKfunctions_pyess-0.0.1-py3-none-any.whl`   
* This will install the TKfunctions as a package within your current python environment  

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

