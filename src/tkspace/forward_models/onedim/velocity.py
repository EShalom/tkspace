#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing forwards model and time step allocation functions for simulations
in the tissue concentration picture in 1D.

In one ocmpartment systems these models describe evolution of tissue concentration, 
which is controlled by blood velocity, diffusion and any sources/sinks within the system.

$$ \\frac{\\partial C} {\\partial t}=
-\\vec{\\nabla} \\cdot \\vec {u} C
+\\vec{\\nabla} \\cdot D^{\\star} \\vec{\\nabla} C $$

Generally for two compartments, these model describe the evolution of arterial and venous tissue concentration, \(C^{a}\) and \(C^{v}\), respectively. 
These are impacted by velocities, \(u\), Psudeo diffusion, \(D\), exchange rate between the compartments, \(K^{va}\).

$$\\frac{\\partial C^{a}}{\\partial t}=
-\\vec{\\nabla} \\cdot \\vec{u}^{a} C^{a}
+\\vec{\\nabla} \\cdot D^{\\star a} \\vec{\\nabla} C^{a}
-K^{va}C^{a}$$

$$
\\frac{\\partial C^{v}}{\\partial t}=
-\\vec{\\nabla} \\cdot \\vec{u}^{v} C^{a}
+\\vec{\\nabla} \\cdot D^{\\star v} \\vec{\\nabla} C^{v}
+K^{va}C^{a}$$

With the measurement concentration being described as:

$$ C = C^{a} + C^{v}$$

This module casts these equations in terms of the positive 
direction flux and negative direction flux:

$$ \\frac{\\partial C^{a}}{\\partial t}=
-(\\nabla j_{p}-\\nabla j_{n})  -K^{va}C^{a}$$


"""

def minmod(a,b):
    
    '''
    Returns minimum of a and b provided they have the same sign
    '''
    
    if a*b < 0:
        return 0
    if abs(a)<abs(b):
        return a
    if abs(b) < abs(a):
        return b

def maxmod(a,b):
    '''
    Returns maximum of a and b provided they have the same sign
    '''
    if a*b < 0:
        return 0
    if abs(a) > abs(b):
        return a
    if abs(b) > abs(a):
        return b

def vanAlbeda1(a,b):
    '''
    Returns Van Albeda 1 limited slope for a and b.
    '''    
    if a*b<=0:
        slope = 0
    if a*b>0:
        slope = (a*b*(a+b))/(a**2+b**2)

    return slope

def superbee(a,b):
    '''
    Returns superbee limited slope for a and b.
    '''
    sig1 = minmod(b,2*a)
    sig2 = minmod(a, 2*b)
    
    return maxmod(sig1, sig2)

def slope_lim(C,Cp,Cn):
    '''
    Inputs:
        - C : voxel concentrations.  
        - Cp : Positive boudary concentration.  
        - Cn : Negative boundary concentration.  
        
    Returns: left and right interface concentrations according to the slope limiter.
    '''
    import numpy as np
    
    grad_l = C[1:-1]-C[:-2]
    grad_r = C[2:]-C[1:-1]
    slope = np.zeros_like(grad_l)
    for i in range(0,len(C)-2):
        slope[i] = vanAlbeda1(grad_l[i],grad_r[i])
            
    grad_pbl = 2*(C[0]-Cp)
    grad_pbr = C[1]-C[0]
    slope_pb = vanAlbeda1(grad_pbl,grad_pbr)
        
    grad_nbr = 2*(Cn-C[-1])
    grad_nbl = C[-1]-C[-2]
    
    slope_nb = vanAlbeda1(grad_nbl,grad_nbr)
        
    C0_left = C[0]-0.5*slope_pb
    C0_right = C[0]+0.5*slope_pb
    CN_left = C[-1]-0.5*slope_nb
    CN_right = C[-1]+0.5*slope_nb
    
    C_left = C[1:-1]-0.5*slope
    C_left = np.append(np.append(C0_left,C_left),CN_left)

    C_right = C[1:-1]+0.5*slope
    C_right = np.append(np.append(C0_right,C_right),CN_right)
    
    return C_left, C_right

def jgrad_onedim(Cprev, u, D, Jp_bnd, Jn_bnd, dx):
    
    '''Calculation of the difference in gradient between of positive and 
    negative flux.
    
    Parameters:  
        
    - Cprev (numpy.ndarray float64):  
        - Concentration (mmol/L) of system at the previous time step.
    - u (numpy.ndarray float64):  
        - Velocity (cm/s) at each interface.  
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
    
    Kp = ((u*H(u,0))/dx + D/dx**2)
    Kn = ((u*H(-u,0))/dx - D/dx**2)
    
    Jp_interface = Cprev * Kp[1:]
    Jn_interface = Cprev * Kn[:-1]
    
    Jp = append(Jp_bnd, Jp_interface)
    Jn = append(Jn_interface, Jn_bnd)
    
    # convert flux from (mmol/mL/s) to (mmol/mm^2/s)
    j = (Jp + Jn)
    
    diff = (j[:-1]-j[1:])

    return diff

def jgrad_central_onedim(Cprev, u, D, Jp_bnd, Jn_bnd, dx):
    
    '''Calculation of the difference in gradient between of positive and 
    negative flux for centred description.
    
    Parameters:  
        
    - Cprev (numpy.ndarray float64):  
        - Concentration (mmol/L) of system at the previous time step.
    - u (numpy.ndarray float64):  
        - Velocity (cm/s) at each cell centre.  
    - D (numpy.ndarray float64):  
        - Pseudo diffusion (cm/s^2) at every cell centre.  
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
    
    u_int = (u[:-1]+u[1:])/2
    u_int = append(u[0],u_int)
    u_int = append(u_int,u[-1])
    
    D_int = (D[:-1]+D[1:])/2
    D_int = append(D[0],D_int)
    D_int = append(D_int,D[-1])
    
    Kp = ((u_int*H(u_int,0))/dx + D_int/dx**2)
    Kn = ((-u_int*H(-u_int,0))/dx  + D_int/dx**2)
    
    Jp_interface = Cprev * Kp[1:]
    Jn_interface = Cprev * Kn[:-1]
    
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

def onedim_onecomp_vel_diff(Lx, dx, tsamp, dtsim, u, D, Jp, Jn):
   
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
        - u (numpy.ndarray float64):  
            - Velocity (cm/s) at each voxel interface. Length = Nx + 1.   
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
    from tkspace.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import max as npmax
    from numpy import linspace, zeros
    
    # Caluclate new internal timesteps and timepoints
    T = npmax(tsamp)
    Ntsim = int(T/dtsim)+1
    tsim = linspace(0,T,Ntsim)
    
    # Interpolate the influxes (mmol/L/s) to the new time resolution
    Jp_bnd = interpolate(Jp, tsamp, tsim)
    Jn_bnd = interpolate(Jn, tsamp, tsim)
    
    # Allocate empty concentration array
    Nx = int(Lx/dx)
    Ctiss = zeros((Nx,Ntsim))
    
    # Populate timestep outputs
    for i in range(1,Ntsim):
        Ctiss[:,i] = Ctiss[:,i-1] - dtsim * jgrad_onedim(Ctiss[:,i-1], u, D, Jp_bnd[i-1], Jn_bnd[i-1], dx)

    return Ctiss

def onedim_onecomp_step(Cprev, Kp, Kn, Jp, Jn, dt):
    '''Calculate concentration update for the next timestep.
    
    Parameters:
        
        - Cprev (numpy.ndarray float64):  
            - Previous concentration (mM) from previous timestep. Length = Nx.  
        - Kp (numpy.ndarray float64):  
            - Positive direction flux (1/s). Combination of velocity and diffusion effects in forwards (left to right) direction.  
        - Kn (numpy.ndarray float64):  
            - Negative direction flux (1/s). Combination of velocity and diffusion effects in backwards (right to left) direction.  
        - Jp (numpy.ndarray float64):  
            - Positive direction influx (mM/s) of concentration at the system's left boundary. Length = len(tsamp).  
        - Jn (numpy.ndarray float64):  
            - Negative direction influx (mM/s) of concentration at the system's right boundary. Length = len(tsamp).  
        - dt (float64):  
            - Timestep for simulation step (s).
    
    Returns:
        
        - Cupdate (numpy.ndarray float64):  
            - Update concentration (mM) for next timestep into. Length = Nx
    '''
    from numpy import empty_like
    
    Cupdate = empty_like(Cprev)
    
    Jp_interface = Kp[1:] * Cprev
    Jn_interface = Kn[:-1] * Cprev
    
    Cupdate[0] = Cprev[0] + (dt) * (Jp + Jn_interface[1] - Jp_interface[0] - Jn_interface[0] )
    
    Cupdate[1:-1] = Cprev[1:-1] + (dt)*( Jp_interface[:-2] - Jn_interface[1:-1] - Jp_interface[1:-1] + Jn_interface[2:] )
     
    Cupdate[-1] = Cprev[-1] + (dt) * (Jn + Jp_interface[-2] - Jn_interface[-1] - Jp_interface[-1] )

    return Cupdate

def timeres1C_velocity(u, D, dx, dtmin):
    
    '''Allocate stable timestep for the forwards simulation.
    
    Parameters:
        
        - u (numpy.ndarray float64):  
            - Velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - D (numpy.ndarray float64):  
            - Pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - dtmin (float64):  
            - Minimum allowed timestep (s) in simulation.  
        
    Returns:
        
        - dt (float64):  
            - Largest timestep for the stable simulation.   
    '''
    import numpy as np

    maxu = abs(np.max(u))
    maxD = np.max((D))
    if maxu ==0:
        dtu=1/dtmin
    else:
        dtu = 0.7*(dx/maxu)
    if maxD==0:
        dtD=1/dtmin
    else:
        dtD = 0.1*(dx**2/maxD)
    dt = np.min((dtu,dtD))
    return dt

def onedim_twocomp_vel_diff(Lx, dx, tsamp, dtsim, ua, uv, Kva, Da, Dv, Jpa, Jpv, Jna, Jnv):
   
    '''Forward model for tissue concentration picture for 1D 2 compartment systems.
    
    Parameters:
        
        - Nx (int):  
            - Number of voxels in the system.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - tsamp (numpy.ndarray float64):  
            - Sampled time points (s). Must start from 0.
        - dtsim (float64):  
            - Internal simulation timestep (s).  
        - ua (numpy.ndarray float64):  
            - Arterial velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - uv (numpy.ndarray float64):  
            - Venous velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - Kva (numpy.ndarray float64):  
            - Transfer  from artery in vein compartment (1/s). Length = Nx.  
        - Da (numpy.ndarray float64):  
            - Arterial pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Dv (numpy.ndarray float64):  
            - Venous pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Jpa (numpy.ndarray float64):  
            - Positive direction arterial influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jpv (numpy.ndarray float64):  
            - Positive direction venous influx (mol/s) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jna (numpy.ndarray float64):  
            - Negative direction arterial influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Jnv (numpy.ndarray float64):  
            - Negative direction venous influx (mol/s) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  

    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
        - Ca (numpy.ndarray float64):  
            - Arterial concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
        - Cv (numpy.ndarray float64):  
            - Venous concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
      '''
    
    from tkspace.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import max as npmax
    from numpy import linspace, zeros
    from numpy import heaviside as H
    #from tkspace.forward_models.onedim.velocity import jgrad_fluxlimiter
    from tkspace.forward_models.onedim.velocity import onedim_twocomp_step
    # Caluclate new internal timesteps and timepoints
    T = npmax(tsamp)
    Ntsim = int(T/dtsim) + 1
    tsim = linspace(0, T, Ntsim)
    
    # Interpolate the influxes to the new time resolution
    Jpa_bnd = interpolate(Jpa, tsamp, tsim)
    Jna_bnd = interpolate(Jna, tsamp, tsim)
    Jpv_bnd = interpolate(Jpv, tsamp, tsim)
    Jnv_bnd = interpolate(Jnv, tsamp, tsim)

    Kpa = ((ua*H(ua,0))/dx + Da/dx**2)
    Kna= ((-ua*H(-ua,0))/dx + Da/dx**2)
    Kpv= ((uv*H(uv,0))/dx + Dv/dx**2)
    Knv = ((-uv*H(-uv,0))/dx + Dv/dx**2)
    
    # Allocate empty concentration array
    Nx = int(Lx/dx)
    Ca = zeros((Nx, Ntsim))
    Cv = zeros((Nx, Ntsim))
    
    # Populate timestep outputs
    for i in range(1, Ntsim):
# =============================================================================
#         Jva = Ca[:,i-1] * Kva
#         Ca[:,i] = Ca[:,i-1] + dtsim * ( + jgrad_fluxlimiter(Ca[:,i-1], ua, Da, Jpa_bnd[i-1], Jna_bnd[i-1], dx, dtsim) - Jva )
#         Cv[:,i] = Cv[:,i-1] + dtsim * ( + jgrad_fluxlimiter(Cv[:,i-1], uv, Dv, Jpv_bnd[i-1], Jnv_bnd[i-1], dx,dtsim) + Jva )
# =============================================================================
        Ca[:,i], Cv[:,i] = onedim_twocomp_step(Ca[:,i-1], Cv[:,i-1], Kpa, Kpv, Kna, Knv, Kva, Jpa_bnd[i-1], Jpv_bnd[i-1], Jna_bnd[i-1], Jnv_bnd[i-1], dtsim)
    # Calculate tissue concentration for the voxel
    Ctiss = Ca + Cv
    return Ctiss, Ca, Cv

def onedim_twocomp_vel_diff_slopelim(Lx, dx, tsamp, dtsim, ua, uv, Kva, Da, Dv, Cpa, Cpv, Cna, Cnv):
   
    '''2nd order forward model with slope limiter for tissue concentration picture for 1D 2 compartment systems.
    
    Parameters:
        
        - Nx (int):  
            - Number of voxels in the system.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - tsamp (numpy.ndarray float64):  
            - Sampled time points (s). Must start from 0.
        - dtsim (float64):  
            - Internal simulation timestep (s).  
        - ua (numpy.ndarray float64):  
            - Arterial velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - uv (numpy.ndarray float64):  
            - Venous velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - Kva (numpy.ndarray float64):  
            - Transfer  from artery in vein compartment (1/s). Length = Nx.  
        - Da (numpy.ndarray float64):  
            - Arterial pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Dv (numpy.ndarray float64):  
            - Venous pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Cpa (numpy.ndarray float64):  
            - Positive direction arterial influx (mM) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Cpv (numpy.ndarray float64):  
            - Positive direction venous influx (mM) of concentration at the system's left boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Cna (numpy.ndarray float64):  
            - Negative direction arterial influx (mM) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  
        - Cnv (numpy.ndarray float64):  
            - Negative direction venous influx (mM) of concentration at the system's right boundary. Length = len(tsamp) (ground truth) or len(tsamp) - 1 (inversion).  

    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
        - Ca (numpy.ndarray float64):  
            - Arterial concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
        - Cv (numpy.ndarray float64):  
            - Venous concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt).  
      '''
    
    from tkspace.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import max as npmax
    from numpy import linspace, zeros
    from numpy import heaviside as H
    #from tkspace.forward_models.onedim.velocity import jgrad_fluxlimiter
    from tkspace.forward_models.onedim.velocity import onedim_twocomp_step_2ndorder
    # Caluclate new internal timesteps and timepoints
    T = npmax(tsamp)
    Ntsim = int(T/dtsim) + 1
    tsim = linspace(0, T, Ntsim)
    
    Cpa = interpolate(Cpa, tsamp, tsim)
    Cna = interpolate(Cna, tsamp, tsim)
    Cpv = interpolate(Cpv, tsamp, tsim)
    Cnv = interpolate(Cnv, tsamp, tsim)
    
    Jpa_bnd = Cpa*((ua[0]*H(ua[0],0))/dx + Da[0]/dx**2)
    Jna_bnd= Cna*((-ua[-1]*H(-ua[-1],0))/dx + Da[-1]/dx**2)
    Jpv_bnd= Cpv*((uv[0]*H(uv[0],0))/dx + Dv[0]/dx**2)
    Jnv_bnd = Cnv*((-uv[-1]*H(-uv[-1],0))/dx + Dv[-1]/dx**2)

    Kpa = ((ua*H(ua,0))/dx + Da/dx**2)
    Kna= ((-ua*H(-ua,0))/dx + Da/dx**2)
    Kpv= ((uv*H(uv,0))/dx + Dv/dx**2)
    Knv = ((-uv*H(-uv,0))/dx + Dv/dx**2)
    
    # Allocate empty concentration array
    Nx = int(Lx/dx)
    Ca = zeros((Nx, Ntsim))
    Cv = zeros((Nx, Ntsim))
    
    # Populate timestep outputs
    for i in range(1, Ntsim):
        Ca_left, Ca_right = slope_lim(Ca[:,i-1], Cpa[i-1], Cna[i-1])
        Cv_left, Cv_right = slope_lim(Cv[:,i-1], Cpv[i-1], Cnv[i-1])
        Ca[:,i], Cv[:,i] = onedim_twocomp_step_2ndorder(Ca[:,i-1],Ca_left, Ca_right, Cv[:,i-1],Cv_left, Cv_right, Kpa, Kpv, Kna, Knv, Kva, Jpa_bnd[i-1], Jpv_bnd[i-1], Jna_bnd[i-1], Jnv_bnd[i-1], dtsim)
    
    # Calculate tissue concentration for the voxel
    Ctiss = Ca + Cv
    return Ctiss, Ca, Cv

def onedim_twocomp_step(Caprev, Cvprev, Kpa, Kpv, Kna, Knv, Kva, Jpa, Jpv, Jna, Jnv, dt):
    
    '''Calculate concentration update for the next timestep.
    
    Parameters:
        
        - Caprev (numpy.ndarray float64):  
            - Previous arterial concentration (mM) from previous timestep. Length = Nx.  
        - Cvprev (numpy.ndarray float64):  
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
        
        - Caupdate (numpy.ndarray float64):  
            - Update arterial concentration (mM) for next timestep into. Length = Nx.  
        - Cvupdate (numpy.ndarray float64):  
            - Update venous concentration (mM) for next timestep into. Length = Nx.  
    '''
    from numpy import empty_like
    
    Caupdate = empty_like(Caprev)
    Cvupdate = empty_like(Cvprev)
# =============================================================================
#     Jpa = Cpa * Kpa[0]
#     Jna = Cna * Kna[-1]
#     Jpv = Cpv * Kpv[0]
#     Jnv = Cnv * Knv[-1]
# =============================================================================
    Jpa_interface = Kpa[1:] * Caprev
    Jna_interface = Kna[:-1] * Caprev
    Jpv_interface = Kpv[1:] * Cvprev
    Jnv_interface = Knv[:-1] * Cvprev
    Jva = Kva * Caprev

    Caupdate[0] = Caprev[0] + (dt) * (Jpa + Jna_interface[1] - Jpa_interface[0] - Jna_interface[0] - Jva[0])
    Cvupdate[0] = Cvprev[0] + (dt) * (Jpv + Jnv_interface[1] - Jpv_interface[0] - Jnv_interface[0] + Jva[0]) 
    
    Caupdate[1:-1]=Caprev[1:-1] + (dt)*( Jpa_interface[:-2] - Jna_interface[1:-1] - Jpa_interface[1:-1] + Jna_interface[2:] - Jva[1:-1])
    Cvupdate[1:-1]=Cvprev[1:-1] + (dt)*( Jpv_interface[:-2] - Jnv_interface[1:-1] - Jpv_interface[1:-1] + Jnv_interface[2:] + Jva[1:-1])

    Caupdate[-1] = Caprev[-1] + (dt) * (Jna + Jpa_interface[-2] - Jna_interface[-1] - Jpa_interface[-1] - Jva[-1] )
    Cvupdate[-1] = Cvprev[-1] + (dt) * (Jnv + Jpv_interface[-2] - Jnv_interface[-1] - Jpv_interface[-1] + Jva[-1] )
    
    return Caupdate, Cvupdate

def onedim_twocomp_step_2ndorder(Caprev,Caprev_l,Caprev_r, Cvprev,Cvprev_l,Cvprev_r, Kpa, Kpv, Kna, Knv, Kva, Jpa, Jpv, Jna, Jnv, dt):
   
    '''Calculate slope limited concentration update for the next timestep.
    
    Parameters:
        
        - Caprev (numpy.ndarray float64):  
            - Previous arterial concentration (mM) from previous timestep. Length = Nx.  
        - Caprev_l (numpy.ndarray float64):  
            - Previous arterial concentration (mM) at each voxels left interface from previous timestep. Length = Nx.  
        - Caprev_r (numpy.ndarray float64):  
            - Previous arterial concentration (mM) at each voxels right interface from previous timestep. Length = Nx.  
        - Cvprev (numpy.ndarray float64):  
            - Previous venous concentration (mM) from previous timestep. Length = Nx.  
        - Cvprev_l (numpy.ndarray float64):  
            - Previous venous concentration (mM) at each voxels left interface from previous timestep. Length = Nx.  
        - Cvprev_r (numpy.ndarray float64):  
            - Previous venous concentration (mM) at each voxels right interface from previous timestep. Length = Nx.  
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
        
        - Caupdate (numpy.ndarray float64):  
            - Update arterial concentration (mM) for next timestep into. Length = Nx.  
        - Cvupdate (numpy.ndarray float64):  
            - Update venous concentration (mM) for next timestep into. Length = Nx.  
    '''    
    from numpy import empty_like
    
    Caupdate = empty_like(Caprev_r)
    Cvupdate = empty_like(Cvprev_r)

    Jpa_interface = Kpa[1:] * Caprev_r
    Jna_interface = Kna[:-1] * Caprev_l
    Jpv_interface = Kpv[1:] * Cvprev_r
    Jnv_interface = Knv[:-1] * Cvprev_l
    
    Jva = Kva * Caprev

    Caupdate[0] = Caprev[0] + (dt) * (Jpa + Jna_interface[1] - Jpa_interface[0] - Jna_interface[0] - Jva[0])
    Cvupdate[0] = Cvprev[0] + (dt) * (Jpv + Jnv_interface[1] - Jpv_interface[0] - Jnv_interface[0] + Jva[0]) 
    
    Caupdate[1:-1]=Caprev[1:-1] + (dt)*( Jpa_interface[:-2] - Jna_interface[1:-1] - Jpa_interface[1:-1] + Jna_interface[2:] - Jva[1:-1])
    Cvupdate[1:-1]=Cvprev[1:-1] + (dt)*( Jpv_interface[:-2] - Jnv_interface[1:-1] - Jpv_interface[1:-1] + Jnv_interface[2:] + Jva[1:-1])

    Caupdate[-1] = Caprev[-1] + (dt) * (Jna + Jpa_interface[-2] - Jna_interface[-1] - Jpa_interface[-1] - Jva[-1] )
    Cvupdate[-1] = Cvprev[-1] + (dt) * (Jnv + Jpv_interface[-2] - Jnv_interface[-1] - Jpv_interface[-1] + Jva[-1] )
    
    return Caupdate, Cvupdate

def timeres2C_velocity(ua, uv, Da, Dv, Kva, dx, dtmin):
    
    '''Allocate stable timestep for the two compartment forwards simulation.
    
    Parameters:
        
        - ua (numpy.ndarray float64):  
            - Arterial velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - uv (numpy.ndarray float64):  
            - Venous velocity (cm/s) at each voxel interface. Length = Nx + 1.  
        - Da (numpy.ndarray float64):  
            - Arterial pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Dv (numpy.ndarray float64):  
            - Venous pseudo diffusion (cm^2/s) at each voxel interface. Length = Nx + 1.  
        - Kva (numpy.ndarray float64):  
            - Transfer  from artery in vein compartment (1/s). Length = Nx.  
        - dx (float64):  
            - Voxel width (cm) in x direction (flow direction).  
        - dtmin (float64):  
            - Minimum allowed timestep (s) in simulation.  
        
    Returns:
        
        - dt (float64):  
            - Largest timestep for the stable simulation.   
    '''
    import numpy as np

    maxua = abs(np.max(ua))
    maxuv = abs(np.max(uv))
    maxDa = np.max((Da))
    maxDv = np.max((Dv))
    maxu = np.max((maxua,maxuv))
    maxD = np.max((maxDa,maxDv))
    maxKva = abs(np.max(Kva))
    
    if maxu ==0:
        dtu=1/dtmin
    else:
        dtu = 0.7*(dx/maxu)
    if maxD==0:
        dtD=1/dtmin
    else:
        dtD = 0.1*(dx**2/maxD)
    if maxKva==0:
        dtKva=1/dtmin
    else:
        dtKva = 0.7/maxKva
    dt = np.min((dtu,dtD,dtKva))
    return dt
