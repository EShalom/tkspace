#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to save different case parameters for 1 dimensional systems.
"""

def case_1D2C(case, res): # Case1: dual aif
    
    '''Save case for 1d2c system type parameters.
    
    Parameters:
        
        - None
    
    Returns:
        
        - Print message with location of produced .npz file.
        
    '''
    
    from TKfunctions.forward_models.onedim.systeminfo import geometry
    import numpy as np
    from TKfunctions.forward_models.onedim.fm_utilities import Ftofafv_1d2c
    from TKfunctions.forward_models.onedim.fm_utilities import parker2006AIF, parker2006AIF_flexible
    from scipy.signal.windows import gaussian as gaussian
    from sigfig import round as sfround
    
    from TKfunctions.utilities import mkdir_p as mkdir
    fdir = 'case_dictionaries/'
    mkdir(fdir)
    
    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry()

    dx, dy, dz = res
    Lx, Ly, Lz, T = sysdim
    dxyz  = dx*dy*dz
    
    fmax, umax, Dmax, psDmax, Fmax, Kvamax, Jmax= max_vals
    
    Nx = int(Lx/dx)
    xc = np.linspace(dx/2,Lx-dx/2,Nx)
    
    if case == 1:
        va_frac = (0.3*(np.cos(xc/100)**2)+0.3)
        v =(0.6*(0.4*np.sin(0.3*xc/10)**2+0.6*np.cos(0.15*xc/10)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 9 # mL/s/mm^2
        fv1 = -5
        F = gaussian(len(xc),len(xc)/2, sym=True)*(0.0626) # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        ua = np.zeros_like(fa)
        uv = np.zeros_like(fv)
        
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
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa = parker2006AIF(tmin[0:-delay])*(abs(ua[0])/dx)
        Jpa_f = parker2006AIF(tmin[0:-delay])*(abs(fa[0])/dx)
        Jpa = np.append(np.zeros(delay),Jpa)
        Jpa_f = np.append(np.zeros(delay),Jpa_f)
        Jpa*=0.6
        Jpa_f*=0.6
        delay = int(10/dt)
        Jna = parker2006AIF(tmin[0:-delay])*(abs(ua[-1])/dx)#*dxyz*1e-9
        Jna_f = parker2006AIF(tmin[0:-delay])*(abs(fa[-1])/dx)
        Jna = np.append(np.zeros(delay),Jna)
        Jna_f = np.append(np.zeros(delay),Jna_f)
        Jna *= 0.4
        Jna_f *= 0.4
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))
        
        fdir = 'case_dictionaries/case1_1d2c'
    if case == 6:
        va_frac = (0.3*(np.cos(xc/100)**2)+0.3)
        v =(0.6*(0.4*np.sin(0.3*xc/10)**2+0.6*np.cos(0.15*xc/10)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 9 # mL/s/mm^2
        fv1 = -5
        F = gaussian(len(xc),len(xc)/2, sym=True)*(0.0626) # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        ua = np.zeros_like(fa)
        uv = np.zeros_like(fv)
        
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
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa = parker2006AIF(tmin[0:-delay])
        Jpa_f = parker2006AIF(tmin[0:-delay])
        Jpa = np.append(np.zeros(delay),Jpa)
        Jpa_f = np.append(np.zeros(delay),Jpa_f)
        Jpa*=0.6
        Jpa_f*=0.6
        delay = int(10/dt)
        Jna = parker2006AIF(tmin[0:-delay])#*dxyz*1e-9
        Jna_f = parker2006AIF(tmin[0:-delay])
        Jna = np.append(np.zeros(delay),Jna)
        Jna_f = np.append(np.zeros(delay),Jna_f)
        Jna *= 0.4
        Jna_f *= 0.4
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))
        
        fdir = 'case_dictionaries/case6_1d2c'
    if case == 5:
        va_frac = (0.3*(np.cos(xc/10)**2)+0.3)
        v =(0.6*(0.4*np.sin(0.3*xc)**2+0.6*np.cos(0.15*xc)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 0.9 # mL/s/mm^2
        fv1 = -0.5
        F = gaussian(len(xc),len(xc)/2, sym=True)*(0.0626) # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        ua = fa/np.append(va[0],va)
        uv = fv/np.append(vv[0],vv)
        Kva = F/va
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa = parker2006AIF(tmin[0:-delay])*(abs(ua[0])/dx)*dxyz*1e-3
        Jpa = np.append(np.zeros(delay),Jpa)
        Jpa*=0.6
        delay = int(10/dt)
        Jna = parker2006AIF(tmin[0:-delay])*(abs(ua[0])/dx)*dxyz*1e-3
        Jna = np.append(np.zeros(delay),Jna)
        Jna *= 0.6
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))
        
        fdir = 'case_dictionaries/case5_1d2c'
    if case == 4:
        va_frac = (0.3*(np.cos(xc/10)**2)+0.3)
        v =(0.6*(0.4*np.sin(0.3*xc)**2+0.6*np.cos(0.15*xc)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 0.9 # mL/s/mm^2
        fv1 = -0.5
        F = gaussian(len(xc),len(xc)/2, sym=True)*(0.0626) # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        fa_centre = (fa[:-1]+fa[1:])/2
        fv_centre = (fv[:-1]+fv[1:])/2
        ua = fa_centre/va
        uv = fv_centre/vv
        Kva = F/va
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa_args = np.array((0.6*0.809, 0.6*0.330, 0.17046, 0.365, 0.0563, 0.132, 1.050, 0.1685, 38.078, 0.483,(abs(ua[0])/dx)*dxyz*1e-3)) #τ = center of the sigmoid
        Jna_args = np.array((0.4*0.809, 0.4*0.330, 0.17046, 0.365, 0.0563, 0.132, 1.050, 0.1685, 38.078, 0.483,(abs(ua[-1])/dx)*dxyz*1e-3)) #τ = center of the sigmoid
        Jpa = parker2006AIF_flexible(tmin[0:-delay],Jpa_args)
        Jpa = np.append(np.zeros(delay),Jpa)
        delay = int(15/dt)
        Jna = parker2006AIF_flexible(tmin[0:-delay],Jna_args)
        Jna = np.append(np.zeros(delay),Jna)
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))
        
        fdir = 'case_dictionaries/case4_1d2c'
    if case == 2:
        va_frac = ( 0.3*(np.cos(0.015*xc)**2)+0.3)
        v =(0.6*(0.5*np.sin(0.01*xc)**2+0.6*np.cos(0.02*xc)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 11 # mL/s/mm^2
        fv1 = -9
        F = (0.6*(0.8*np.cos(0.01*xc)**2)+0.35)*0.07 # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        ua = np.zeros_like(fa)
        uv = np.zeros_like(fv)
        
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
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa = parker2006AIF(tmin[0:-delay])
        Jpa = np.append(np.zeros(delay),Jpa)
        Jna = np.zeros(Nt+1)
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        Jna_f = Jna
        Jpa_f = Jpa
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))

        
        fdir = 'case_dictionaries/case2_1d2c'
    elif case == 3:
        va_frac = ( 0.3*(np.sin(0.007*xc)**2)+0.3)
        v =(0.6*(0.5*np.sin(0.02*xc)**2+0.6*np.cos(0.03*xc)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 3 # mL/s/mm^2
        fv1 = -6
        F = (0.6*(0.8*np.sin(2000*xc)**2)+0.3)*0.07 # mL/s/mL
        fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
        
        ua = np.zeros_like(fa)
        uv = np.zeros_like(fv)
        
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
        
        dtD = 0.1*(dx**2/Dmax)
        dtu = 0.7*(dx/umax)
        dt = min(dtD, dtu)
        dt = sfround(dt,1)
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt+1)
        tmin = t/60
        delay = int(10/dt)
        Jpa = parker2006AIF(tmin[0:-delay])
        Jpa = np.append(np.zeros(delay),Jpa)
        delay = int(15/dt)
        Jna = parker2006AIF(tmin[0:-delay])
        Jna = np.append(np.zeros(delay),Jna)
        Jpv = np.zeros(Nt+1)
        Jnv = np.zeros(Nt+1)
        Jna_f = Jna
        Jpa_f = Jpa
        
        P = np.append(ua,uv)
        P = np.append(P,Kva)
        P = np.append(P,Da)
        P = np.append(P,Dv)
        P = np.append(P,Jpa)
        P = np.append(P,Jpv)
        P = np.append(P,Jna)
        P = np.append(P,Jnv)
        
        Pmax = np.tile(umax,ua.size+uv.size)
        Pmax = np.append(Pmax,np.tile(Kvamax,Kva.size))
        Pmax = np.append(Pmax,np.tile(Dmax,Da.size+Dv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpa.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jpv.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jna.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jnv.size))
        
        fdir = 'case_dictionaries/case3_1d2c'
    
    np.savez_compressed(fdir, fa = fa, fv=fv, F=F, va = va, va_frac=va_frac, vv=vv, v=v, ua = ua, uv = uv, Da = Da, Dv = Dv, Kva = Kva, Jpa = Jpa, Jpv = Jpv, Jna = Jna, Jnv = Jnv,Jpa_f = Jpa_f, Jna_f = Jna_f, t =t, Nt = Nt, dt = dt, P = P, Pmax = Pmax)
    
    msg = '.npz output at {}'.format(fdir)
    
    return msg

def case1_1D1C(): # Case1: dual aif
    
    '''Save case 1 for 1d2c system type parameters.
    
    Parameters:
        
        - None
    
    Returns:
        
        - Print message with location of produced .npz file.
        
    '''
    
    from TKfunctions.forward_models.onedim.systeminfo import geometry
    import numpy as np
    from TKfunctions.forward_models.onedim.fm_utilities import Ftofafv_1d2c
    from TKfunctions.forward_models.onedim.velocity import timeres1C_velocity
    from TKfunctions.forward_models.onedim.fm_utilities import parker2006AIF
    from TKfunctions.utilities import tidy
    from scipy.signal.windows import gaussian as gaussian

    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry()

    dx, dy, dz = meas_res
    Lx, Ly, Lz, T = sysdim
    dxyz, dxz, dyz = dx*dy*dz, dx*dz, dy*dz
    
    fmax, umax, Dmax, psDmax, Fmax, Kvamax, Cmax = max_vals
    
    Nx, Ny, Nz = meas_voxels
    xc = np.linspace(dx/2,Lx-dx/2,Nx)
    
    v =(0.6*(0.4*np.sin(0.3*xc/10)**2+0.6*np.cos(0.15*xc/10)**2)+0.3)
    
    D = np.zeros(Nx+1)
    
    f = np.ones((Nx+1))*5 # mL/s/mm^2
    u = np.zeros_like(f)
    
    for i in range(0,len(f)):
            if f[i] < 0:
                if i == len(f)-1:
                    u[i] = f[i]/v[i-1]
                else:
                    u[i] = f[i]/v[i]
            elif f[i] > 0:
                if i == 0:
                    u[i] = f[i]/v[i]
                else:
                    u[i] = f[i]/v[i-1]
            else:
                u[i]=0
    
    dtD = 0.1*(dx**2/Dmax)
    dtu = 0.7*(dx/umax)
    dt = min(dtD, dtu)
    dt=0.1
    
    Nt = int(T/dt)
    
    t=np.linspace(0,T,Nt+1)
    tmin = t/60
    delay = int(10/dt)
    Cl = parker2006AIF(tmin[0:-delay])
    Cl = np.append(np.zeros(delay),Cl)
    Cr = np.zeros(Nt+1)

    
    P = np.append(u,D)
    P = np.append(P,Cl)
    P = np.append(P,Cr)
    
    Pmax = np.tile(umax,u.size)
    Pmax = np.append(Pmax,np.tile(Dmax,D.size))
    Pmax = np.append(Pmax,np.tile(Cmax,Cl.size))
    Pmax = np.append(Pmax,np.tile(Cmax,Cr.size))
    
    fdir = 'case_dictionaries/case1_1d1c'
    
    np.savez_compressed(fdir, f=f, v=v, u = u, D = D, Cl = Cl, Cr = Cr, t =t, Nt = Nt, dt = dt, P = P, Pmax = Pmax)
    
    msg = '.npz output at {}'.format(fdir)
    
    return msg
