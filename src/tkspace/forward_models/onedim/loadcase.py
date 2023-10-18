#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions with load parameters for specifc saved cases based
on system type, case number, dimension and picture.
"""

def load_P(dims,comp,case):
    
    '''Load the ground truth parameter list from a specific file.
    
    Parameters:
        
        - dims (int):   
            - Dimension number.  
        - comp (int)   
            - Number of compartments.  
        - case (int):   
            - Case number within the system type.  
    
    Returns:
        - Pnorm (numpy.ndarray float64):   
            - Normalised parameter list for the initial guess in the optimisation.  
        - Pmax (numpy.ndarray float64):   
            - Maximum parameter value list.  
        - dt (float64):   
            - Time resolution (s) of loaded parameters.  
        - Nt (int):   
            - Number of time points.  
        - t (numpy.ndarray float64):   
            - Array of timepoints (s).  
        '''
    import numpy as np
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c.npz'.format(case,dims,comp)
    
    params = np.load('{}{}'.format(fdir,fname),allow_pickle=True)
    
    t = params['t']
    Nt = params['Nt']
    dt = params['dt']
    P = params['P']
    Pmax = params['Pmax']
    
    Pnorm = P/Pmax
    
    return Pnorm, Pmax, dt, Nt, t

def load_Pguess(dims,comp,pic,case,guess):
    
    '''Load the guess parameter list from a specific file.
    
    Parameters:
        
        - dims (int):  
            - Dimension number.  
        - comp (int):   
            - Number of compartments.  
        - pic (int):   
            - Picture representation type. 1: Local, 2: Tissue.  
        - case (int):   
            - Case number within the system type.  
        - guess (int):   
            - Guess number to use.  
v    
    Returns:
        - Pinitial (numpy.ndarray float64):   
            - Parameter list for the initial guess in the optimisation.  
        - Pmax (numpy.ndarray float64):   
            - Maximum parameter value lis.  
        - Pfix (numpy.ndarray float64):   
            - Value deciding if a parameter will be optimised in algorithm. Fixed = 0, Free = 1.  
        - Pbnds (nested list):   
            - List of pairs of each lower and upper bound for each parameter.  
    '''
    
    import numpy as np
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c_pic{}.npz'.format(case,dims,comp,pic)
    
    params = np.load('{}{}'.format(fdir,fname))
    
    Pinitial = params['Pinitial']
    Pmax = params['Pmax']
    Pfix = params['Pfix']
    Pbnds = params['Pbnds']
    
    return Pinitial, Pmax, Pfix, Pbnds
    
def load_velocity_1d1c(dims,comp,case):
    
    '''Load the physical parameters from a specific file that uses the tissue picture.
    
    Parameters:
        - dims (int):  
            - Dimension number.  
        - comp (int)   
            - Number of compartments.   
        - case (int):   
            - Case number within the system type.  
        
    Returns:
        - u (numpy.ndarray float64):   
            - Velocity (cm/s) at each voxel interface. Length = Nx + 1.    
        - D (numpy.ndarray float64):   
            - Pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jl (numpy.ndarray float64):   
            - Influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jr (numpy.ndarray float64):   
            - Influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Nt (int):   
            - Number of timepoints.
        - dt (float64):   
            - Time resolution (s).
    '''
    
    import numpy as np
    
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c.npz'.format(case,dims,comp)
    
    params = np.load('{}{}'.format(fdir,fname))
    
    u = params['u']
    D = params['D']
    Jl = params['Jl']
    Jr = params['Jr']
    Nt = params['Nt']
    dt = params['dt']
    t = params['t']
    
    return u, D, Jl, Jr, dt, Nt, t

def load_flow_1d1c(dims,comp,case):
    
    '''Load the physical parameters from a specific file that uses the local picture.
    
    Parameters:
        - dims (int):  
            - Dimension number.  
        - comp (int)   
            - Number of compartments.   
        - case (int):   
            - Case number within the system type.  
        
    Returns:
        - f (numpy.ndarray float64):   
            - Flow (ml/s/cm^2) at each voxel interface. Length = Nx + 1.    
        - D (numpy.ndarray float64):   
            - Pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - v (numpy.ndarray float64):  
            - Volume fraction of compartment in voxel (mL/mL)
        - Jl (numpy.ndarray float64):   
            - Influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jr (numpy.ndarray float64):   
            - Influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Nt (int):   
            - Number of timepoints.
        - dt (float64):   
            - Time resolution (s).
    '''
    
    import numpy as np
    
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c.npz'.format(case,dims,comp)
    
    params = np.load('{}{}'.format(fdir,fname),allow_pickle=True)
    
    f = params['f']
    D = params['D']
    v = params['v']
    Jl = params['Jl']
    Jr = params['Jr']
    Nt = params['Nt']
    dt = params['dt']
    t = params['t']
    
    return f, D, v, Jl, Jr, dt, Nt, t

def load_velocity_1d2c(dims,comp,case):
    
    '''Load the physical parameters from a specific file that uses the tissue picture.
    
    Parameters:
        - dims (int):  
            - Dimension number.  
        - comp (int)   
            - Number of compartments.  
        - pic (int):   
            - Picture representation type. 1: Local, 2: Tissue.  
        - case (int):   
            - Case number within the system type.  
        
    Returns:
        - ua (numpy.ndarray float64):   
            - Arterial velocity (cm/s) at each voxel interface. Length = Nx + 1.    
        - uv (numpy.ndarray float64):   
            - Venous velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - Kva (numpy.ndarray float64):   
            - Transfer  from artery in vein compartment (1/s). Length = N.  
        - Da (numpy.ndarray float64):   
            - Arterial pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Dv (numpy.ndarray float64):   
            - Venous pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jla (numpy.ndarray float64):   
            - Arterial influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jlv (numpy.ndarray float64):   
            - Venous influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jra (numpy.ndarray float64):   
            - Arterial influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jrv (numpy.ndarray float64):   
            - Venous influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Nt (int):   
            - Number of timepoints.
        - dt (float64):   
            - Time resolution (s).
    '''
    
    import numpy as np
    
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c.npz'.format(case,dims,comp)
    
    params = np.load('{}{}'.format(fdir,fname))
    
    ua = params['ua']
    uv = params['uv']
    Da = params['Da']
    Dv = params['Dv']
    Kva = params['Kva']
    Jpa = params['Jpa']
    Jpv = params['Jpv']
    Jna = params['Jna']
    Jnv = params['Jnv']
    Nt = params['Nt']
    dt = params['dt']
    t = params['t']
    
    return ua, uv, Da, Dv, Kva, Jpa, Jpv, Jna, Jnv, dt, Nt, t

def load_flow_1d2c(dims,comp,case):
    
    '''Load the physical parameters from a specific file that uses the tissue picture.
    
    Parameters:
        - dims (int):  
            - Dimension number.  
        - comp (int)   
            - Number of compartments.  
        - pic (int):   
            - Picture representation type. 1: Local, 2: Tissue.  
        - case (int):   
            - Case number within the system type.  
        
    Returns:
        - ua (numpy.ndarray float64):   
            - Arterial velocity (cm/s) at each voxel interface. Length = Nx + 1.    
        - uv (numpy.ndarray float64):   
            - Venous velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - Kva (numpy.ndarray float64):   
            - Transfer  from artery in vein compartment (1/s). Length = N.  
        - Da (numpy.ndarray float64):   
            - Arterial pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Dv (numpy.ndarray float64):   
            - Venous pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jla (numpy.ndarray float64):   
            - Arterial influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jlv (numpy.ndarray float64):   
            - Venous influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jra (numpy.ndarray float64):   
            - Arterial influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jrv (numpy.ndarray float64):   
            - Venous influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Nt (int):   
            - Number of timepoints.
        - dt (float64):   
            - Time resolution (s).
    '''
    
    import numpy as np
    
    from tkspace.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    fname = 'case{}_{}d{}c.npz'.format(case,dims,comp)
    
    params = np.load('{}{}'.format(fdir,fname),allow_pickle=True)
    
    fa = params['fa']
    fv = params['fv']
    Da = params['Da']
    Dv = params['Dv']
    F = params['F']
    va = params['va']
    va_frac = params['va_frac']
    vv = params['vv']
    v = params['v']
    Jpa = params['Jpa_f']
    Jpv = params['Jpv']
    Jna = params['Jna_f']
    Jnv = params['Jnv']
    Nt = params['Nt']
    dt = params['dt']
    t = params['t']
    
    return fa, fv, Da, Dv, F, va, va_frac, vv, v, Jpa, Jpv, Jna, Jnv, dt, Nt, t
