#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module conatining functions for calculating cost function values.
"""

def rms_psp_cost(Precon, Pmax, args):
    
    ''' Caluclates the average root mean sqaured (rms) difference between the current guess concentration curves and the ground truth measurement per sampling point (psp).
    
    Parameters:
        
        - Precon (numpy.ndarray float64):    
            - Current parameter set in the optimisation.  
        - args (Tuple):    
            - Collection of extra arguements used by forwards model.  
    
    Returns:
        
        - cost (float64):    
            - Mean difference between the current guess and the ground truth per sampling point.
    '''
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    Cmeas, fmod, fmod_setup, optres, sysdim, tsamp, compartments, dtsim, pic,step,dx = args
    
    if compartments == 1:
        Crecon = fmod_setup(fmod, Precon, Pmax, dtsim, optres, sysdim, tsamp, pic, 1) 
    if compartments == 2:
        Crecon, Ca, Cv = fmod_setup(fmod, Precon, Pmax,dtsim, optres, sysdim, tsamp, pic, 1)  
    
    # If simulation returns answer which 'blows up' returns cost function 1 to prevent errors in interpolation
    if np.isnan(Crecon).any()==True:
        return 1
    
    t = np.linspace(0, sysdim[-1], np.shape(Crecon)[-1])
    
    Dt = tsamp[1]-tsamp[0]
    skip = int(Dt/dtsim)
    
    
    Cinv= Crecon[:,0::skip]
    diff = (Cmeas)-(Cinv)
    #diff = np.divide(diff, Cmeas, out=None, where=Cmeas!=0)
    cost = (1/(np.array(Cinv.shape).prod(axis=0)))*abs(np.sqrt(np.sum(np.power(diff,2))))

    return cost

def rms_psp_cost_scipy(Precon, args):
    
    ''' Caluclates the average root mean sqaured (rms) difference between the current guess concentration curves and the ground truth measurement per sampling point (psp).
    
    Parameters:
        
        - Precon (numpy.ndarray float64):    
            - Current parameter set in the optimisation.  
        - args (Tuple):    
            - Collection of extra arguements used by forwards model.  
    
    Returns:
        
        - cost (float64):    
            - Mean difference between the current guess and the ground truth per sampling point.
    '''
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    Cmeas, fmod, fmod_setup, optres, sysdim, tsamp, compartments, dtsim, pic,Pmax = args
    
    if compartments == 1:
        Crecon = fmod_setup(fmod, Precon, Pmax, dtsim, optres, sysdim, tsamp, pic, 1) 
    if compartments == 2:
        Crecon, Ca, Cv = fmod_setup(fmod, Precon, Pmax,dtsim, optres, sysdim, tsamp, pic, 0)  
    
    # If simulation returns answer which 'blows up' returns cost function 1 to prevent errors in interpolation
    if np.isnan(Crecon).any()==True:
        return 1
    
    t = np.linspace(0, sysdim[-1], np.shape(Crecon)[-1])
    
    Dt = tsamp[1]-tsamp[0]
    skip = int(Dt/dtsim)
    
    
    Cinv= Crecon[:,0::skip]
    diff = Cmeas-Cinv
    diff = np.divide(diff, Cmeas, out=None, where=Cmeas!=0)
    cost = (1/(np.array(Cinv.shape).prod(axis=0)))*abs(np.sqrt(np.sum(np.power(diff,2))))

    return cost
