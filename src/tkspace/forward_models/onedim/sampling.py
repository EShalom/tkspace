#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions for temporal down or upsampling, addition of noise and 
conversion between lcoal and tissue concentrations. 
"""

def downsampler(C, T, dt, Dt, usesamp,samplet):
    '''Downsamples a concnetration array in time using provides timesteps, or a provided sampling time array.
    
    Parameters:
        
        - C(numpy.ndarray float64):  
            - Original concentration at full time resolution dt.  
        - T (float64):  
            - End point for sampling.  
        - dt (float64):  
            - Original time resolution (s) of concentration array.  
        - Dt (float64):  
            - New time resolution (s) of concentration array.  
        - usesamp (int):  
            - Use given sample times. Yes=1: Uses samplet as sampling times. No=0: Allocates sampling time using Dt and random generated starting time.  
        - samplet (numpy.ndarray float64):  
            - Sampling time array. Must be an array if usesamp = 1.  
    
    Returns:
        
        - Cmeas (numpy.ndarray float64):  
            -  Concentration (mM) at measurement sampling points.  
        - tsamp (numpy.ndarray float64):  
            - Sampling time array used for Cmeas. 
    '''
    import numpy as np
    from scipy.interpolate import interp1d as interpol
    
    Nt_og = C.shape[-1]
    t = np.linspace(0,T,Nt_og)
    
    if usesamp == 1:
        tsamp = samplet
        Nt_new = len(tsamp)
        Nx = C.shape[0]
        Cmeas = np.zeros((Nx, Nt_new))
        for i in range(0,Nx):
            Cmeas[i,:]= interpol(t, C[i,:], kind='linear')(tsamp)
        
    if usesamp == 0:
        skip = int(Dt/dt)
        tsamp = t[0::skip]
        Cmeas = C[:,0::skip]
    
    return Cmeas, tsamp

def gaussian_noise(Cmeas, CNR):
    ''' Adds guassian noise with standard deviation defined by the Contrast-to-Noise Ratio provided.
    
    Parameters:
        
        - Cmeas (numpy.ndarray float64):  
            - Measurement concentration (mM) to have noise added to.  
        - CNR (float64):  
            - Contrast to Noise Ratio of noise to add.  
    
    Returns:
        
        - Cmeas OR Cnoisy (numpy.ndarray float64):  
            - Dependent on CNR if 0 return same array, in non-zero adds rician noise and return the noisy data array.
    '''
    import numpy as np
    
    if CNR == 0:
        print("No noise added \n")
        return Cmeas
    
    stdev = np.mean(Cmeas)/CNR    
    noise = np.random.normal(loc=0.0,scale=stdev, size=np.shape(Cmeas))
    Cnoisy = Cmeas+noise
    Cnoisy = abs(Cnoisy) # Ensure noise pattern in Rician
    return Cnoisy




