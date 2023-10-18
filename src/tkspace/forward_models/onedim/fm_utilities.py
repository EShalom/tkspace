#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions such as extractfing physical parameters and 
using them to run through forwards simulations.
"""

def interp_linear(values, old_points, new_points):
    
    ''' Linear interpolation from coarse resolution to fine resolution.'''
    
    from scipy.interpolate import interp1d
    
    interp_values = interp1d(old_points, values, kind = 'linear', bounds_error=False,fill_value=(values[0],values[-1]))(new_points)
    
    return interp_values
    
def fmodsetup_1d1c(fmod, Pnorm, Pmax, dtsim, res, sysdim, tsamp, pic, inv):
    
    ''' Extracts parameters from stack and runs in correct forward model.
    
    Parameters:
        
        - fmod (callable function):  
             - Forwards model which returns concentration array.  
        - Pnorm (numpy.ndarray float64):  
             - Normalised parameter stack, values 0 <= abs(1).  
        - Pmax (numpy.ndarray float64):  
            - Maximum parameter stack, to convert back to unnormalised quantities.  
        - res (list):  
            - Spatial resolution and minimum time resolution of the system for simulation. res = (dx,dy,dz,dtmin).  
        - sysdim (list):  
            - System total length in space and time. sysdim = (Lx, Ly, Lz, T).  
        - tsamp (numpy.ndarray float64):  
            - Sampling timepoints (s).  
        - inv (0,1):  
            - Option to run simulation as ground truth or inversion. Last time point is interpolated if inversion option selected.  
    
    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt)  
    
    '''
    
    from numpy import heaviside as H
    from numpy import append  
    dx, dy, dz = res
    Lx, Ly, Lz, T = sysdim
    
    Nx = int(Lx/dx)
    P = Pnorm * Pmax
    
    if pic == 1:
        f, D, v, Cp, Cn = param_extract_flow_1d1c(P, Nx)
        Jp = Cp * ((f*H(f,0))/dx + D[0]/dx**2)
        Jn = Cn * ((-f*H(-f,0))/dx + D[-1]/dx**2)
    elif pic == 2:
        u, D, Jp, Jn = param_extract_vel_1d1c(P, Nx)
        Jp *= H(u[0],0)
        Jn *= H(-u[-1],0)
    if inv == 1:
        Jp = append(Jp,Jp[-1])
        Jn = append(Jn,Jn[-1])
    
    if pic == 1:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, f, D, v, Jp, Jn)
    elif pic == 2:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, u, D, Jp, Jn)
    
    return C_tiss

def fmodsetup_1d2c(fmod, Pnorm, Pmax, dtsim, res, sysdim, tsamp, pic, inv):
    
    ''' Extracts parameters from stack and runs in correct forward model.
    
    Parameters:
        
        - fmod (callable function):  
            - Forwards model which returns concentration array.  
        - Pnorm (numpy.ndarray float64):  
            - Normalised parameter stack, values 0 <= abs(1).  
        - Pmax (numpy.ndarray float64):  
            - Maximum parameter stack, to convert back to unnormalised quantities.  
        - res (list):  
            - Spatial resolution and minimum time resolution of the system for simulation. res = (dx,dy,dz,dtmin).  
        - sysdim (list):  
            - System total length in space and time. sysdim = (Lx, Ly, Lz, T).  
        - tsamp (numpy.ndarray float64):  
            - Sampling timepoints (s).  
        - inv (0,1):  
            - Option to run simulation as ground truth or inversion. Last time point is interpolated if inversion option selected.  
    
    Returns:
        
        - Ctiss (numpy.ndarray float64):  
            - Concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt)  
        - Ca (numpy.ndarray float64):  
            - Arterial concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt)  
        - Cv (numpy.ndarray float64):  
            - Venous concentration (mM) in the system in space and time. Shape = (1,Nx,1,Nt)  
    '''
    
    from numpy import heaviside as H
    from numpy import append
    from tkspace.forward_models.onedim.fm_utilities import parker2006AIF_flexible as aif
    dx, dy, dz = res
    dxyz = dx*dy*dz
    Lx, Ly, Lz, T = sysdim
    
    Nx = int(Lx/dx)
    P = Pnorm * Pmax
    
    if pic == 1:
        fa1, fv1,fa,fv, F, Da, Dv, va, va_frac, vv,v, Jpa, Jpv, Jna, Jnv = param_extract_flow_1d2c(P, Nx,dx)
        Jpa *= H(fa[0],0)
        Jna *= H(-fa[-1],0)
        Jpv *= H(fv[0],0)
        Jnv *= H(-fv[-1],0)
    elif pic == 2:
        ua, uv, Kva, Da, Dv, Jpa, Jpv, Jna, Jnv = param_extract_vel_1d2c(P, Nx)
        Jpa *= H(ua[0],0)
        Jna *= H(-ua[-1],0)
        Jpv *= H(uv[0],0)
        Jnv *= H(-uv[-1],0)
    elif pic == 4:
        ua, uv, Kva, Da, Dv, Cpa, Cpv, Cna, Cnv = param_extract_vel_1d2c(P, Nx)
        Jpa = Cpa*((ua[0]*H(ua[0],0))/dx + Da[0]/dx**2)
        Jna = Cna*((-ua[-1]*H(-ua[-1],0))/dx + Da[-1]/dx**2)
        Jpv = Cpv*((uv[0]*H(uv[0],0))/dx + Dv[0]/dx**2)
        Jnv = Cnv*((-uv[-1]*H(-uv[-1],0))/dx + Dv[-1]/dx**2)
        Jpa *= H(ua[0],0)
        Jna *= H(-ua[-1],0)
        Jpv *= H(uv[0],0)
        Jnv *= H(-uv[-1],0)
    elif pic == 5:
        fa1, fv1,fa,fv, F, Da, Dv, va, va_frac, vv,v, Cpa, Cpv, Cna, Cnv = param_extract_flow_1d2c(P, Nx, dx)
        Jpa = Cpa*((fa[0]*H(fa[0],0))/dx + Da[0]/dx**2)
        Jna = Cna*((-fa[-1]*H(-fa[-1],0))/dx + Da[-1]/dx**2)
        Jpv = Cpv*((fv[0]*H(fv[0],0))/dx + Dv[0]/dx**2)
        Jnv = Cnv*((-fv[-1]*H(-fv[-1],0))/dx + Dv[-1]/dx**2)
        Jpa *= H(fa[0],0)
        Jna *= H(-fa[-1],0)
        Jpv *= H(fv[0],0)
        Jnv *= H(-fv[-1],0)
    elif pic == 3:
        ua, uv, Kva, Da, Dv, Ja, Jpa_frac, Jv, Jpv_frac = param_extract_vel_1d2c_msres(P, Nx)
        Jpa = Ja * Jpa_frac
        Jna = Ja - Jpa
        Jpv = Jv * Jpv_frac
        Jnv = Jv - Jpv
        Jpa *= H(ua[0],0)
        Jna *= H(-ua[-1],0)
        Jpv *= H(uv[0],0)
        Jnv *= H(-uv[-1],0)
        Jpa = append(Jpa,Jpa[-1])
        Jna = append(Jna,Jna[-1])
        Jpv = append(Jpv,Jpv[-1])
        Jnv = append(Jnv,Jnv[-1])
    if inv == 1:
        Jpa = append(Jpa,Jpa[-1])
        Jna = append(Jna,Jna[-1])
        Jpv = append(Jpv,Jpv[-1])
        Jnv = append(Jnv,Jnv[-1])
    if inv == 2:
        Jpa = aif(Jpa)
        Jna = aif(Jna)
        Jpv = aif(Jpv)
        Jnv = aif(Jnv)
    
    # Convert into concentration (mmol/s --> mmol/mL/s)
# =============================================================================
#     Jpa = Jpa / (dxyz)
#     Jna = Jna / (dxyz)
#     Jpv = Jpv / (dxyz)
#     Jnv = Jnv / (dxyz)
# =============================================================================
    
    if pic == 1:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, fa, fv, F, Da, Dv, va, vv, Jpa, Jpv, Jna, Jnv)

    elif pic == 2:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, ua, uv, Kva, Da, Dv, Jpa,Jpv, Jna, Jnv)
    elif pic == 3:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, ua, uv, Kva, Da, Dv, Jpa,Jpv, Jna, Jnv)
    elif pic == 4:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, ua, uv, Kva, Da, Dv, Jpa,Jpv, Jna, Jnv)
    elif pic == 5:
        C_tiss = fmod(Lx, dx, tsamp, dtsim, fa, fv, F, Da, Dv, va, vv, Jpa, Jpv, Jna, Jnv)
    
    
    return C_tiss

def Ftofafv_1d2c(F,fa1,fv1,dx):
    
    '''Calculate arterial and venous flow using incompressiblity constraint.
    Uses exchange term F = - d/dx(fa) = + d/dx(fv) to define the total flow of the system as incompressible.
    Make use of boundary values of flows.
     
    Parameters:
        
        - F (numpy.ndarray float64):  
            - Mono-directional exchange term from arterial to venous compartment. (mL/mL).  
        - fa1 (float64):  
            - Left boundary value of the arterial flow. (mL/s/cm^2).  
        - fv1 (float64):  
            - Left boundary value of the venous flow. (mL/s/cm^2).  
    
    Returns:
        - fa (numpy.ndarray float64):  
            - Arterial flow at interfaces. (mL/s/cm^2).          
        - fv (numpy.ndarray float64):  
            - Venous flow at interfaces. (mL/s/cm^2).  
    '''
    
    from numpy import empty
    # Uses exchange term F = - d/dx(fa) = + d/dx(fv) to define the
    # 
    
    fa = empty(len(F)+1)
    fv = empty(len(F)+1)
    fa[0] = fa1
    fv[0] = fv1

    for i in range(1,len(F)+1):
        fa[i]=(-F[i-1]*dx)+fa[i-1]
        fv[i]=(F[i-1]*dx)+fv[i-1]
    
    return fa, fv

def param_extract_flow_1d1c(P, Nx):
    
    '''Extracts parameters from stack for local picture in 1d1c.'''
    
    Nt = int((len(P)-2*(Nx+1))/2)
    f = P[0]
    D = P[1:Nx+2]
    v = P[Nx+2:2*Nx+2]
    Jl = P[2*Nx+2:2*Nx+2+Nt]
    Jr = P[2*Nx+2+Nt:]
    
    output = [f, D, v, Jl, Jr]
    return output

def param_extract_flow_1d2c(P, Nx,dx):
    
    '''Extracts parameters from stack for local picture in 1d2c.'''
    
    Nt = int(((len(P)-(2*(Nx+1)+3*Nx+2))/4))
    fa1 = P[0]
    fv1 = P[1]
    F   = P[2:Nx+2]
    Da = P[(Nx+1)+1:2*(Nx+1)+1]
    Dv = P[2*(Nx+1)+1:3*(Nx+1)+1]
    va_frac = P[3*(Nx+1)+1:4*(Nx+1)]
    v = P[4*(Nx+1):5*(Nx+1)-1]
    Bla = P[5*(Nx+1)-1:5*(Nx+1)-1+Nt]
    Blv = P[5*(Nx+1)-1+Nt:5*(Nx+1)-1+2*Nt]
    Bra = P[5*(Nx+1)-1+2*Nt:5*(Nx+1)-1+(3*Nt)]
    Brv = P[5*(Nx+1)-1+3*Nt:5*(Nx+1)-1+(4*Nt)]
    
    va = v*va_frac
    vv=v-va
    fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx )    
    output = [fa1, fv1,fa,fv, F, Da, Dv, va, va_frac,vv, v, Bla,Blv, Bra, Brv]
    return output

def param_extract_vel_1d1c(P, Nx):
    
    '''Extracts parameters from stack for tissue picture in 1d1c.'''
    
    Nt = int((len(P)-2*(Nx+1))/2)
    u = P[0:Nx+1]
    D = P[Nx+1:2*(Nx+1)]
    Jl = P[2*Nx+2:2*Nx+2+Nt]
    Jr = P[2*Nx+2+Nt:2*Nx+2+2*Nt]
    
    output = [u, D, Jl, Jr]
    return output

def param_extract_vel_1d2c(P, Nx):
    
    '''Extracts parameters from stack for tissue picture in 1d2c.'''
    
    Nt = int(((len(P)-(4*(Nx+1)+Nx))/4))
    ua = P[0:Nx+1]
    uv = P[Nx+1:2*(Nx+1)]
    Kva  = P[2*(Nx+1):2*(Nx+1)+Nx]
    Da = P[2*(Nx+1)+Nx:3*(Nx+1)+Nx]
    Dv = P[3*(Nx+1)+Nx:4*(Nx+1)+Nx]
    Bla = P[4*(Nx+1)+Nx:4*(Nx+1)+Nx+Nt]
    Blv = P[4*(Nx+1)+Nx+Nt:4*(Nx+1)+Nx+2*Nt]
    Bra = P[4*(Nx+1)+Nx+2*Nt:4*(Nx+1)+Nx+3*Nt]
    Brv = P[4*(Nx+1)+Nx+3*Nt:4*(Nx+1)+Nx+4*Nt]
    
    output = [ua, uv, Kva, Da, Dv, Bla, Blv, Bra, Brv]
    return output


def param_extract_vel_1d2c_msres(P,Nx):
    
    '''Extracts parameters from stack for multiresolution tissue picture in 1d2c.'''
    
    Nt = int(((len(P)-(4*(Nx+1)+Nx))/4))
    ua = P[0:Nx+1]
    uv = P[Nx+1:2*(Nx+1)]
    Kva  = P[2*(Nx+1):2*(Nx+1)+Nx]
    Da = P[2*(Nx+1)+Nx:3*(Nx+1)+Nx]
    Dv = P[3*(Nx+1)+Nx:4*(Nx+1)+Nx]
    Ba = P[4*(Nx+1)+Nx:4*(Nx+1)+Nx+Nt]
    Bpa_frac = P[4*(Nx+1)+Nx+Nt:4*(Nx+1)+Nx+2*Nt]
    Bv = P[4*(Nx+1)+Nx+2*Nt:4*(Nx+1)+Nx+3*Nt]
    Bpv_frac = P[4*(Nx+1)+Nx+3*Nt:4*(Nx+1)+Nx+4*Nt]
    
    output = [ua, uv, Kva, Da, Dv, Ba, Bpa_frac, Bv, Bpv_frac]
    return output

def parker2006AIF(t):
    
    '''Produces an arterial input function (AIF) based on the findings in (Parker, 2006).
    
    Parameters:
        
        - t (numpy.ndarray float64):  
            - Timepoints to calculate AIF over (mins).
    
    Returns:
        
        - aif (numpy.ndarray float64):  
            - AIF over timepoints supplied in mM.
    '''
    import numpy as np
    
    A1 = 0.809 # An = scaling constantsof the nth Gaussian
    A2 = 0.330
    T1 = 0.17046 # Tn = centers of the nth Gaussian
    T2 = 0.365
    sig1 = 0.0563 # σn = widths of the nth Gaussian
    sig2 = 0.132 
    alpha = 1.050 #α = amplitude of the exponential  
    beta = 0.1685 # β = decay constant of the exponential 
    s = 38.078 #s = width of the sigmoid
    tau = 0.483 #τ = center of the sigmoid
    
    c1 = (A1/(sig1*np.sqrt(2*np.pi)))*np.exp(-np.power((t-T1),2)/(2*np.power(sig1,2)))+((alpha*np.exp(-beta*t))/(1+np.exp(-s*(t-tau))))
    c2 = (A2/(sig2*np.sqrt(2*np.pi)))*np.exp(-np.power((t-T2),2)/(2*np.power(sig2,2)))+((alpha*np.exp(-beta*t))/(1+np.exp(-s*(t-tau))))
    aif = c1+c2 # cb = population averaged AIF
    
    return aif

def parker2006AIF_flexible(t,params):
    
    '''Produces an arterial input function (AIF) based on the findings in (Parker, 2006).
    
    Parameters:
        
        - t (numpy.ndarray float64):  
            - Timepoints to calculate AIF over (mins).
    
    Returns:
        
        - aif (numpy.ndarray float64):  
            - AIF over timepoints supplied in mM.
    '''
    import numpy as np
    
    A1 = params[0]
    A2 = params[1]
    T1 = params[2]
    T2  = params[3] 
    sig1 = params[4]
    sig2 = params[5]
    alpha= params[6] 
    beta= params[7] 
    s= params[8] 
    tau= params[9]
    scale = params[10]
    
    c1 = (A1/(sig1*np.sqrt(2*np.pi)))*np.exp(-np.power((t-T1),2)/(2*np.power(sig1,2)))+((alpha*np.exp(-beta*t))/(1+np.exp(-s*(t-tau))))
    c2 = (A2/(sig2*np.sqrt(2*np.pi)))*np.exp(-np.power((t-T2),2)/(2*np.power(sig2,2)))+((alpha*np.exp(-beta*t))/(1+np.exp(-s*(t-tau))))
    aif = c1+c2 # cb = population averaged AIF
    
    return scale*aif



def convertTtoL(value_tuple):
    
    from numpy import zeros_like
    
    fa1, fv1,fa,fv, F, Da, Dv, va, va_frac,vv, v, Bla,Blv, Bra, Brv = value_tuple
    
    ua = zeros_like(fa)
    uv = zeros_like(fv)
    
    for i in range(0,len(fa)):
        if fa[i] < 0:
            if i == len(fa)-1:
                ua[i] = fa[i]/va[i-1]
            else:
                ua[i] = fa[i]/va[i]
        elif fa[i] > 0:
            if i == 0:
                ua[i] = fa[i]/va[i]
            else:
                ua[i] = fa[i]/va[i-1]
        else:
            ua[i]=0
        
        if fv[i] < 0:
            if i == len(fv)-1:
                uv[i] = fv[i]/vv[i-1]
            else:
                uv[i] = fv[i]/vv[i]
        elif fv[i] > 0:
            if i == 0:
                uv[i] = fv[i]/vv[i]
            else:
                uv[i] = fv[i]/vv[i-1]
        else:
            uv[i]=0

    Kva = F/va
    
    output = [ua, uv, Kva, Da, Dv, Bla, Blv, Bra, Brv]
    
    return output

def convertTtoL_oneComp(value_tuple):
    
    from numpy import zeros_like
    
    f1, D, v, Bl, Br = value_tuple
    
    u = zeros_like(D)
    
    for i in range(0,len(D)):
        if f1 < 0:
            if i == len(D)-1:
                u[i] = f1/v[i-1]
            else:
                u[i] = f1/v[i]
        elif f1 > 0:
            if i == 0:
                u[i] = f1/v[i]
            else:
                u[i] = f1/v[i-1]
        else:
            u[i]=0

    output = [u, D, Bl, Br]
    
    return output






