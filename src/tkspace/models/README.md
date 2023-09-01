## Overview

---

This package contains sub packages for one dimension forward models. 
These forward models evolve tracer concentration according to spatiotemporal 
tracer kinetic equations in [(Sourbron, 2014)](https://ieeexplore.ieee.org/document/6716985).

These model fall into two representation types:  

- The tissue concentration picture models. These are cast in terms of velocity 
and pseudo diffuion.  
    - Found in `velocity`
- The local conentration picture can be found in the flow module. These are 
cast in terms of flow and volume fraction.  
    - Found in `flow`  
    
All models take in physical parameters as arguements and produce a tissue 
concentration array (accessible by measurement) as the output. Time 
downsampling and noise addition can also be applied and is found in the 
corresponding sampling module.  

## Usage

---

See the examples directory for python scripts of demonstrations on:  

- Use of one dimension one comprtment forward model: ex_onedimonecomp.py.  
- Use of one dimension two comprtment forward model: ex_onedimtwocomp.py.  
- Use of whole routine with forward simsulation and inversion recovery for one
dimensional systems: package_example.py.  
