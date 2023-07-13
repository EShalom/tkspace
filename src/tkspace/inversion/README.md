## Overview

---

This package contains modules containing optimisation algorithms for the
recovery phsyical transport quantities from spatiotemporal meaurements of  
concentration. These algorithms are written to take in normalised parameters
(max $\\leq$ 1) and and use appropriate forward models and cost evaluations 
provided by the user.

## Modules

---

This package consists of:  

- A guess alloctaion module to provide an initial guess set of parameters which 
matches the system type required.    
    - Found in `guess`  
- A gradient descent module with various funstions for this local search type.  
    - Found in `gradient_descent`  
- A cost functions module for caluclation of the value for optimisation.  
    - Found in `cost_functions`  
    
All models take in physical parameters as arguements and produce a tissue 
concentration array (accessible by measurement) as the output. Time 
downsampling and noise addition can also be applied and is found in the 
corresponding sampling module.  

## Usage

---

See the examples directory for python scripts of demonstrations on:  
  
- Use of whole routine with forward simsulation and inversion recovery for one
dimensional systems: package_example.py.  
