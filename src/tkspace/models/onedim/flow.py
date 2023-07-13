#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing forwards model and time step allocation functions for simulations
in the local concentration picture in 1D.

In one compartment systems these models describe evolution of local concentration, 
which is controlled by blood flow, diffusion and any sources/sinks within the system.

$$ v \\frac{\\partial c} {\\partial t}=
-\\vec{\\nabla} \\cdot ( \\vec {f} c )
+\\vec{\\nabla} \\cdot ( D \\vec{\\nabla} c ) $$

Generally for two compartments, these model describe the evolution of arterial and venous tissue concentration, \(C^{a}\) and \(C^{v}\), respectively. 
These are impacted by flows, \(f\), molecular diffusion, \(D\), exchange rate between the compartments, \(F\).

$$v^{a} \\frac{\\partial c^{a}}{\\partial t}=
-\\vec{\\nabla} \\cdot ( \\vec{f}^{a} c^{a} )
+\\vec{\\nabla} \\cdot ( D^{a} \\vec{\\nabla} c^{a} )
-F^{a}$$

$$ v^{v} \\frac{\\partial c^{v}}{\\partial t}=
-\\vec{\\nabla} \\cdot ( \\vec{f}^{v} c^{a} )
+\\vec{\\nabla} \\cdot ( D^{v} \\vec{\\nabla} c^{v} )
+Fc^{a}$$

With the measurement concentration being described as:

$$ C = v^{a}c^{a} + v^{v}c^{v}$$

This module casts these equations in terms of the positive 
direction flux and negative direction flux:

$$ v^{a}\\frac{\\partial c^{a}}{\\partial t}=
-(\\nabla j_{p}-\\nabla j_{n})  -Fc^{a}$$


"""

def jgrad_onedim(cprev, f, D, Jp_bnd, Jn_bnd, dx):
    
    '''Calculation of the difference in gradient between of positive and 
    negative flux.
    
    Parameters:  
        
    - cprev (numpy.ndarray float64):  
        - Concentration (mmol/L) of system at the previous time step.
    - f (numpy.ndarray float64):  
        - Flow (ml/s/cm^2) at each interface.  
    - D (numpy.ndarray float64):  
        - Pseudo diffusion (cm/s^2) at every interface.  
    - Jp_bnd (numpy.ndarray float64): 
        - Positive direction influx (mmol/L/s) of concentration at the system's left boundary.  
        - Length = len(tsamp).  
    - Jn_bnd (numpy.ndarray float64):  
        - Negative direction influx (mmol/L/s) of concentration at the system's right boundary.  
        - Length = len(tsamp).  
    - dx (float64):  
        - Voxel width (cm) in x direction (flow direction).
    
    Returns:  
        
    - diffJ (numpy.ndarray float64):  
        - Difference in gradient between of positive and negative flux. (mmol/L/s) 
    '''
    
    from numpy import heaviside as H
    from numpy import append
    
    Kp = ((f*H(f,0))/dx + D/dx**2)
    Kn = ((-f*H(-f,0))/dx  + D/dx**2)
    
    Jp_interface = cprev * Kp[1:]
    Jn_interface = cprev * Kn[:-1]
    
    Jp = append(Jp_bnd, Jp_interface)
    Jn = append(Jn_interface, Jn_bnd)
    
    # convert flux from (mmol/L/s) to (mmol/cm^2/s)
    jp = Jp * dx * 1e-3
    jn = Jn * dx * 1e-3
    
    gradjp = (jp[1:]-jp[:-1])/dx
    gradjn = (jn[1:]-jn[:-1])/dx
    
    diffJ =( - gradjn + gradjp)
    
    diffJ = diffJ * 1e3 # convert mmol/ml/s to mmol/L/s
    
    return diffJ

def onedim_onecomp_flow_diff(Lx, dx, tsamp, dtsim, f, D, v, Jp, Jn):
   
    '''Forward model for tissue concentration picture for 1D 1 compartment systems.
    
    Parameters:
        
        - Lx (int):  
            - System length in the x direction.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - tsamp (numpy.ndarray float64):  
            - Sampled time points (s). Must start at 0.  
        - dtsim (float64):  
            - Internal simulation timestep (s).  
        - f (numpy.ndarray float64):  
            - Flow (cm/s) at each voxel interface. Length = Nx + 1.   
        - D (numpy.ndarray float64):  
            - Pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jp (numpy.ndarray float64):  
            - Positive direction influx (mM/s) of concentration at the system's left boundary. Length = len(tsamp).  
        - Jn (numpy.ndarray float64):  
            - Negative direction influx (mM/s) of concentration at the system's right boundary. Length = len(tsamp).  
    
    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (Nx,Nt).  
            
    '''
    from TKfunctions.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import max as npmax
    from numpy import linspace, zeros, ones
    from numpy import heaviside as H
    # Caluclate new internal timesteps and timepoints
    T = npmax(tsamp)
    Ntsim = int(T/dtsim)+1
    tsim = linspace(0,T,Ntsim)
    
    # Interpolate the influxes (mmol/L/s) to the new time resolution
    Jp_bnd = interpolate(Jp, tsamp, tsim)
    Jn_bnd = interpolate(Jn, tsamp, tsim)
    
    # Allocate empty concentration array
    Nx = int(Lx/dx)
    c_loc = zeros((Nx,Ntsim))
    
    f = ones((len(D)))*f
    Kp = ((f*H(f,0))/dx + D/dx**2)
    Kn = ((-f*H(-f,0))/dx + D/dx**2)
    
    # Populate timestep outputs
    for i in range(1,Ntsim):
        #c_loc[:,i] = c_loc[:,i-1] - (dtsim/v) * jgrad_onedim(c_loc[:,i-1], f, D, Jp_bnd[i-1], Jn_bnd[i-1], dx)
        c_loc[:,i] = onedim_onecomp_step(c_loc[:,i-1], v, Kp, Kn, Jp_bnd[i-1], Jn_bnd[i-1], dtsim)
        
    C_tiss = localtotissue(c_loc, v)
    return C_tiss

def onedim_onecomp_step(cprev, v, Kp, Kn, pos_bnd,neg_bnd, dt):
    
    '''Calculate concentration update for the next timestep.
    
    Parameters:
        
        - caprev (numpy.ndarray float64):  
            - Previous arterial concentration (mM) from previous timestep. Length = Nx.  
        - cvprev (numpy.ndarray float64):  
            - Previous venous concentration (mM) from previous timestep. Length = Nx.  
        - Kpa (numpy.ndarray float64):  
            - Positive direction arterial flux (1/s). Combination of velocity and diffusion effects in forwards (left to right) direction.  
        - Kpv (numpy.ndarray float64):  
            - Positive direction venous flux (1/s). Combination of velocity and diffusion effects in forwards (left to right) direction.  
        - Kna (numpy.ndarray float64):  
            - Negative direction arterial flux (1/s). Combination of velocity and diffusion effects in backwards (right to left) direction.  
        - Knv (numpy.ndarray float64):  
            - Negative direction venous flux (1/s). Combination of velocity and diffusion effects in backwards (right to left) direction.  
        - Kav (numpy.ndarray float64):  
            - Transfer constant (1/s).  
        - Jpa (numpy.ndarray float64):  
            - Positive direction arterial influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jpv (numpy.ndarray float64):  
            - Positive direction venous influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jna (numpy.ndarray float64):  
            - Negative direction arterial influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jnv (numpy.ndarray float64):  
            - Negative direction venous influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - dt (float64):  
            - Timestep for simulation step (s).
    
    Returns:
        
        - caupdate (numpy.ndarray float64):  
            - Update arterial concentration (mM) for next timestep into. Length = Nx.  
        - cvupdate (numpy.ndarray float64):  
            - Update venous concentration (mM) for next timestep into. Length = Nx.  
    '''
    from numpy import empty_like
    
    cupdate = empty_like(cprev)
    Jp_interface = Kp[1:] * cprev
    Jn_interface = Kn[:-1] * cprev

    cupdate[0] = cprev[0] + (dt/v[0]) * (pos_bnd + Jn_interface[1] - Jp_interface[0] - Jn_interface[0])
    
    cupdate[1:-1]=cprev[1:-1] + (dt/v[1:-1])*( Jp_interface[:-2] - Jn_interface[1:-1] - Jp_interface[1:-1] + Jn_interface[2:])

    cupdate[-1] = cprev[-1] + (dt/v[-1]) * (neg_bnd + Jp_interface[-2] - Jn_interface[-1] - Jp_interface[-1])

    return cupdate

def onedim_twocomp_flow_diff(Lx, dx, tsamp, dtsim, fa, fv, F, Da, Dv, va, vv, Jpa, Jpv, Jna, Jnv):
   
    '''Forward model for tissue concentration picture for 1D 1 compartment systems.
    
    Parameters:
        
        - Lx (int):  
            - System length in the x direction.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - tsamp (numpy.ndarray float64):  
            - Sampled time points (s). Must start at 0.  
        - dtsim (float64):  
            - Internal simulation timestep (s).  
        - f (numpy.ndarray float64):  
            - Flow (cm/s) at each voxel interface. Length = Nx + 1.   
        - D (numpy.ndarray float64):  
            - Pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jp (numpy.ndarray float64):  
            - Positive direction influx (mM/s) of concentration at the system's left boundary. Length = len(tsamp).  
        - Jn (numpy.ndarray float64):  
            - Negative direction influx (mM/s) of concentration at the system's right boundary. Length = len(tsamp).  
    
    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (Nx,Nt).  
            
    '''
    from TKfunctions.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from TKfunctions.forward_models.onedim.flow import onedim_twocomp_step
    from numpy import max as npmax
    from numpy import linspace, zeros
    from numpy import heaviside as H
    
    # Caluclate new internal timesteps and timepoints
    T = npmax(tsamp)
    Ntsim = int(T/dtsim)+1
    tsim = linspace(0,T,Ntsim)

    
    # Interpolate the influxes (mmol/L/s) to the new time resolution
    Jpa_bnd = interpolate(Jpa, tsamp, tsim)
    Jna_bnd = interpolate(Jna, tsamp, tsim)
    Jpv_bnd = interpolate(Jpv, tsamp, tsim)
    Jnv_bnd = interpolate(Jnv, tsamp, tsim)
    
    Kpa = ((fa*H(fa,0))/dx + Da/dx**2)
    Kna= ((-fa*H(-fa,0))/dx + Da/dx**2)
    Kpv= ((fv*H(fv,0))/dx + Dv/dx**2)
    Knv = ((-fv*H(-fv,0))/dx + Dv/dx**2)
    
    # Allocate empty concentration array
    Nx = int(Lx/dx)
    ca_loc = zeros((Nx,Ntsim))
    cv_loc = zeros((Nx,Ntsim))
    
    # Populate timestep outputs
    for i in range(1,Ntsim):
        #ca_loc[:,i] = ca_loc[:,i-1] + (dtsim/va) * ( - jgrad_onedim(ca_loc[:,i-1], fa, Da, Jpa_bnd[i-1], Jna_bnd[i-1], dx) - F*ca_loc[:,i-1])
        #cv_loc[:,i] = cv_loc[:,i-1] + (dtsim/vv) * ( - jgrad_onedim(cv_loc[:,i-1], fv, Dv, Jpv_bnd[i-1], Jnv_bnd[i-1], dx) + F*ca_loc[:,i-1])
        ca_loc[:,i], cv_loc[:,i] = onedim_twocomp_step(ca_loc[:,i-1], cv_loc[:,i-1],va,vv,F, Kpa, Kpv, Kna, Knv, Jpa_bnd[i-1], Jpv_bnd[i-1], Jna_bnd[i-1], Jnv_bnd[i-1], dtsim)

    Ca = localtotissue(ca_loc, va)
    Cv = localtotissue(cv_loc, vv)
    C_tiss = Ca + Cv
    return C_tiss, Ca, Cv

def onedim_twocomp_step(caprev, cvprev, va, vv,F, Kpa, Kpv, Kna, Knv, Jpa, Jpv, Jna, Jnv, dt):
    
    '''Calculate concentration update for the next timestep.
    
    Parameters:
        
        - caprev (numpy.ndarray float64):  
            - Previous arterial concentration (mM) from previous timestep. Length = Nx.  
        - cvprev (numpy.ndarray float64):  
            - Previous venous concentration (mM) from previous timestep. Length = Nx.  
        - Kpa (numpy.ndarray float64):  
            - Positive direction arterial flux (1/s). Combination of velocity and diffusion effects in forwards (left to right) direction.  
        - Kpv (numpy.ndarray float64):  
            - Positive direction venous flux (1/s). Combination of velocity and diffusion effects in forwards (left to right) direction.  
        - Kna (numpy.ndarray float64):  
            - Negative direction arterial flux (1/s). Combination of velocity and diffusion effects in backwards (right to left) direction.  
        - Knv (numpy.ndarray float64):  
            - Negative direction venous flux (1/s). Combination of velocity and diffusion effects in backwards (right to left) direction.  
        - Kav (numpy.ndarray float64):  
            - Transfer constant (1/s).  
        - Jpa (numpy.ndarray float64):  
            - Positive direction arterial influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jpv (numpy.ndarray float64):  
            - Positive direction venous influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jna (numpy.ndarray float64):  
            - Negative direction arterial influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jnv (numpy.ndarray float64):  
            - Negative direction venous influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - dt (float64):  
            - Timestep for simulation step (s).
    
    Returns:
        
        - caupdate (numpy.ndarray float64):  
            - Update arterial concentration (mM) for next timestep into. Length = Nx.  
        - cvupdate (numpy.ndarray float64):  
            - Update venous concentration (mM) for next timestep into. Length = Nx.  
    '''
    from numpy import empty_like
    
    caupdate = empty_like(caprev)
    cvupdate = empty_like(cvprev)
    Jpa_interface = Kpa[1:] * caprev
    Jna_interface = Kna[:-1] * caprev
    Jpv_interface = Kpv[1:] * cvprev
    Jnv_interface = Knv[:-1] * cvprev
    Jva = F * caprev

    caupdate[0] = caprev[0] + (dt/va[0]) * (Jpa + Jna_interface[1] - Jpa_interface[0] - Jna_interface[0] - Jva[0])
    cvupdate[0] = cvprev[0] + (dt/vv[0]) * (Jpv + Jnv_interface[1] - Jpv_interface[0] - Jnv_interface[0] + Jva[0]) 
    
    caupdate[1:-1]=caprev[1:-1] + (dt/va[1:-1])*( Jpa_interface[:-2] - Jna_interface[1:-1] - Jpa_interface[1:-1] + Jna_interface[2:] - Jva[1:-1])
    cvupdate[1:-1]=cvprev[1:-1] + (dt/vv[1:-1])*( Jpv_interface[:-2] - Jnv_interface[1:-1] - Jpv_interface[1:-1] + Jnv_interface[2:] + Jva[1:-1])

    caupdate[-1] = caprev[-1] + (dt/va[-1]) * (Jna + Jpa_interface[-2] - Jna_interface[-1] - Jpa_interface[-1] - Jva[-1] )
    cvupdate[-1] = cvprev[-1] + (dt/vv[-1]) * (Jnv + Jpv_interface[-2] - Jnv_interface[-1] - Jpv_interface[-1] + Jva[-1] )

    return caupdate, cvupdate

def localtotissue(c,v):
    ''' Converts local concentrations into tissue concnetrations which are accessible by measurement.
    
    Parameters:
        
        - c (numpy.ndarray float64):  
            - Local concnetration array. Shape = (1,Nx,1,Nt).  
        - v (numpy.ndarray float64):  
            - Volume fraction arrat. Length = Nx.  
    
    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Tissue concnetration array. Shape = (1,Nx,1,Nt).
    '''
    import numpy as np
    
    Ctiss = np.zeros_like(c)
    
    Ctiss[:,:] = c*np.dstack((np.shape(c)[-1]*(v,)))
    
    return Ctiss

