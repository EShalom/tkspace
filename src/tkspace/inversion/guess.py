#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to allocate initial guess parameter sets
for 1 dimensional systems.
"""
def guess_1d1c(tsamp, case, guess_set):
    
    '''Allocate a guess parameter stack for inversion method.
    
    Parameters:
        
        - tsamp (numpy.ndarray float64):    
            - Sampling time array.  
        - case (int):    
            - Case of the system type this guess is for.  
        - guess_set (int):    
            - Number of the guess being set.  
        
    Returns:
        
        - P (numpy.ndarray float64):    
            - Guess set of parameters.  
        - Pmax (numpy.ndarray float64):    
            - Set of maximum values same length as P.  
        - Pfix (numpy.ndarray int64):    
            - Same length as P. Shows if parameter will be optimised in algorithm. Fixed = 0, Free = 1.  
        - ub (numpy.ndarray float64):    
            - Upper bounds for every parameter in Pinitial.  
        - lb (numpy.ndarray float64):    
            - Lower bounds for every parameter in Pinitial.  
        '''
        
    from TKfunctions.forward_models.onedim.systeminfo import geometry
    from TKfunctions.forward_models.onedim.flow import Ftofafv_1d2c
    import numpy as np

    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry()

    dx, dy, dz, dtmin = meas_res
        
    guess_no, u_fac, D_fac, Jl_fac, Jr_fac = guess_set
        
    Nx, Ny, Nz = meas_voxels
    Nt = len(tsamp)
    
    u = np.ones((Nx+1))*u_fac
    
    D = np.ones(Nx+1)*D_fac
    
    Jl = np.ones(Nt-1)*Jl_fac
    Jr = np.ones(Nt-1)*Jr_fac
    
    P = np.append(u,D)
    P = np.append(P,Jl)
    P = np.append(P,Jr)
    
    Pfix = [1,]*(len(u))
    Pfix += [0,]*(len(D))
    Pfix += [1,]*len(Jl)
    Pfix += [1,]*len(Jr)
    
    Pfix = np.asarray(Pfix)

    
    return P, Pfix

def guess_1d2c(tsamp, case, guess_set):
    
    '''Allocate a guess parameter stack for inversion method.
    
    Parameters:
        
        - tsamp (numpy.ndarray float64):    
            - Sampling time array.  
        - case (int):    
            - Case of the system type this guess is for.  
        - guess_set (int):    
            - Number of the guess being set.  
        
    Returns:
        
        - P (numpy.ndarray float64):    
            - Guess set of parameters.  
        - Pmax (numpy.ndarray float64):    
            - Set of maximum values same length as P.  
        - Pfix (numpy.ndarray int64):    
            - Same length as P. Shows if parameter will be optimised in algorithm. Fixed = 0, Free = 1.  
        - ub (numpy.ndarray float64):    
            - Upper bounds for every parameter in Pinitial.  
        - lb (numpy.ndarray float64):    
            - Lower bounds for every parameter in Pinitial.  
        '''
        
    from TKfunctions.forward_models.onedim.systeminfo import geometry
    from TKfunctions.forward_models.onedim.flow import Ftofafv_1d2c
    import numpy as np

    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry()

    dx, dy, dz, dtmin = meas_res
    dxyz, dxz, dyz = dx*dy*dz, dx*dz, dy*dz
    
    fmax, umax, Dmax, psDmax, Fmax, Kvamax, Jamax, Jvmax, JaTmax, JvTmax = max_vals
    
    guess_no, fa1_fac, fv1_fac, Da_fac, Dv_fac, F_fac, Jla_fac, Jlv_fac, Jra_fac, Jrv_fac, vguess= guess_set
    
    Jamax = JaTmax * dxyz * 1e-3
    Jvmax = JvTmax * dxyz * 1e-3
    
    Nx, Ny, Nz = meas_voxels
    Nt = len(tsamp)
    
    fa1 = fa1_fac # mL/s/mm^2
    fv1 = fv1_fac
    F = np.ones((Nx))*F_fac*Fmax
    fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
    
    Kva = (F/(0.5*vguess))/Kvamax
    
    vguess = np.append(vguess[0],vguess)
    
    ua = (fa/(0.5*vguess))/umax
    uv = (fv/(0.5*vguess))/umax
    
    
    Da = np.ones(Nx+1)*Da_fac
    Dv = np.ones(Nx+1)*Dv_fac
    
    Jla = np.ones(Nt-1)*Jla_fac
    Jlv = np.ones(Nt-1)*Jlv_fac
    Jra = np.ones(Nt-1)*Jra_fac
    Jrv = np.ones(Nt-1)*Jrv_fac
    
    P = np.append(ua,uv)
    P = np.append(P,Kva)
    P = np.append(P,Da)
    P = np.append(P,Dv)
    P = np.append(P,Jla)
    P = np.append(P,Jlv)
    P = np.append(P,Jra)
    P = np.append(P,Jrv)
    
    Pmax = np.tile(umax,ua.size+uv.size)
    Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
    Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
    Pmax = np.append(Pmax,np.tile(Jamax,Jla.size))
    Pmax = np.append(Pmax,np.tile(Jvmax,Jlv.size))
    Pmax = np.append(Pmax,np.tile(Jamax,Jra.size))
    Pmax = np.append(Pmax,np.tile(Jvmax,Jrv.size))
    
    Pfix = [1,]*(len(ua))
    Pfix += [1,]*(len(uv))
    Pfix += [1,]*(len(Kva))
    Pfix += [0,]*(len(Da)+len(Dv))
    Pfix += [1,]*len(Jla)
    Pfix += [0,]*len(Jlv)
    Pfix += [1,]*len(Jra)
    Pfix += [0,]*len(Jrv)
    
    Pfix = np.asarray(Pfix)
    
    Pbnds = [[-1,1],]*(len(ua)+len(uv))
    Pbnds += [[0.001,1],]*(len(Kva))
    Pbnds += [[0,1],]*(len(Da)+len(Dv))
    Pbnds += [[0,1],]*len(Jla)
    Pbnds += [[0,1],]*len(Jlv)
    Pbnds += [[0,1],]*len(Jra)
    Pbnds += [[0,1],]*len(Jrv)
    
    lb = np.asarray(Pbnds)[:,0]
    ub = np.asarray(Pbnds)[:,1]
    
    return P, Pmax, Pfix, ub, lb
