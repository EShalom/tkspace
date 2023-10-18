#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to save different case parameters for 1 dimensional systems.
"""
def case_1D2C_cm(case, res): # Case1: dual aif
    
    '''Save case for 1d2c system type parameters.
    
    Parameters:
        
        - None
    
    Returns:
        
        - Print message with location of produced .npz file.
        
    '''
    
    from tkspace.forward_models.onedim.systeminfo import geometry_cm
    import numpy as np
    from tkspace.forward_models.onedim.fm_utilities import Ftofafv_1d2c
    from tkspace.forward_models.onedim.fm_utilities import parker2006AIF, parker2006AIF_flexible
    from scipy.signal.windows import gaussian as gaussian
    
    from tkspace.utilities import mkdir_p as mkdir

    import math

    def quad(t, K):
        nt = len(t)
        mid = math.floor(nt/2)
        return quadratic(t, t[0], t[mid], t[-1], K[0], K[1], K[2])

    def quadratic(x, x1, x2, x3, y1, y2, y3):
        """returns a quadratic function of x 
        that goes through the three points (xi, yi)"""

        a = x1*(y3-y2) + x2*(y1-y3) + x3*(y2-y1)
        a /= (x1-x2)*(x1-x3)*(x2-x3)
        b = (y2-y1)/(x2-x1) - a*(x1+x2)
        c = y1-a*x1**2-b*x1
        return a*x**2+b*x+c

    fdir = 'case_dictionaries/'
    mkdir(fdir)
    
    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry_cm()

    dx, dy, dz = res
    Lx, Ly, Lz, T = sysdim
    dxyz  = dx*dy*dz
    
    fmax, umax, Dmax, psDmax, Fmax, Kvamax, Jmax= max_vals
    
    Nx = int(Lx/dx)
    xc = np.linspace(dx/2,Lx-dx/2,Nx)
    x = np.linspace(0,Lx,Nx+1)

    Da = np.zeros(Nx+1)
    Dv = np.zeros(Nx+1)

    fa1 = 0.512 # mL/s/cm^2
    fv1 = -0.512
    F = gaussian(len(xc),len(xc)/6, sym=True)*(0.1) # mL/s/mL
    fa, fv = Ftofafv_1d2c(F,fa1,fv1,dx)
    
    va_int = abs(fa)/quad(x,[19,4.9,19])
    vv_int = abs(fv)/quad(x,[7.1,1.5,7.1])
    v_int = va_int + vv_int
    
    ua = fa/va_int
    uv = fv/vv_int
    
    va = (va_int[:-1]+va_int[1:])/2
    vv = (vv_int[:-1]+vv_int[1:])/2
    v = va+vv
    va_frac = va/v

    Kva = F/va
    
    dtD = 0.1*(dx**2/Dmax)
    dtu = 0.7*(dx/umax)
    dt = min(dtD, dtu)
    dt = np.floor(dt*100)/100

    Nt = int(T/dt)
    
    t=np.linspace(0,T,Nt+1)
    tmin = t/60
    delay = int(10/dt)
    Jpa = parker2006AIF(tmin[0:-delay])
    Jpa = np.append(np.zeros(delay),Jpa)
    delay = int(10/dt)
    Jna = parker2006AIF(tmin[0:-delay])#*dxyz*1e-9
    Jna = np.append(np.zeros(delay),Jna)
    Jpv = np.zeros(Nt+1)
    Jnv = np.zeros(Nt+1)
    Jpa_f = Jpa
    Jna_f = Jna

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

    fdir = 'case_dictionaries/case11_1d2c'

    np.savez_compressed(fdir, fa = fa, fv=fv, F=F, va = va, va_frac=va_frac, vv=vv, v=v, ua = ua, uv = uv, Da = Da, Dv = Dv, Kva = Kva, Jpa = Jpa, Jpv = Jpv, Jna = Jna, Jnv = Jnv,Jpa_f = Jpa_f, Jna_f = Jna_f, t =t, Nt = Nt, dt = dt, P = P, Pmax = Pmax)
    
    msg = '.npz output at {}'.format(fdir)
    
    return msg


def case_1D2C(case, res): # Case1: dual aif
    
    '''Save case for 1d2c system type parameters.
    
    Parameters:
        
        - None
    
    Returns:
        
        - Print message with location of produced .npz file.
        
    '''
    
    from tkspace.forward_models.onedim.systeminfo import geometry
    import numpy as np
    from tkspace.forward_models.onedim.fm_utilities import Ftofafv_1d2c
    from tkspace.forward_models.onedim.fm_utilities import parker2006AIF, parker2006AIF_flexible
    from scipy.signal.windows import gaussian as gaussian
    from sigfig import round as sfround
    
    from tkspace.utilities import mkdir_p as mkdir
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
        
        fa1 = 0.9 # mL/s/cm^2
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
        
        fa1 = 0.9 # mL/s/cm^2
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
        
        fa1 = 11 # mL/s/cm^2
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
    if case == 3:
        va_frac = ( 0.3*(np.sin(0.007*xc)**2)+0.3)
        v =(0.6*(0.5*np.sin(0.02*xc)**2+0.6*np.cos(0.03*xc)**2)+0.3)
        va = v*va_frac
        vv = v-va
        
        Da = np.zeros(Nx+1)
        Dv = np.zeros(Nx+1)
        
        fa1 = 3 # mL/s/cm^2
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
    elif case == 10:
        va_frac = (0.1*(np.cos(xc/100)**2)+0.1)
        v =(0.5*(0.4*np.sin(0.3*xc/10)**2+0.5*np.cos(0.15*xc/7)**2)+0.5)
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
        
        fdir = 'case_dictionaries/case10_1d2c'
    
    np.savez_compressed(fdir, fa = fa, fv=fv, F=F, va = va, va_frac=va_frac, vv=vv, v=v, ua = ua, uv = uv, Da = Da, Dv = Dv, Kva = Kva, Jpa = Jpa, Jpv = Jpv, Jna = Jna, Jnv = Jnv,Jpa_f = Jpa_f, Jna_f = Jna_f, t =t, Nt = Nt, dt = dt, P = P, Pmax = Pmax)
    
    msg = '.npz output at {}'.format(fdir)
    
    return msg

def case1_1D1C(case, res): # Case1: dual aif
    
    '''Save case 1 for 1d2c system type parameters.
    
    Parameters:
        
        - None
    
    Returns:
        
        - Print message with location of produced .npz file.
        
    '''
    
    from tkspace.forward_models.onedim.systeminfo import geometry
    import numpy as np
    from tkspace.forward_models.onedim.fm_utilities import Ftofafv_1d2c
    from tkspace.forward_models.onedim.velocity import timeres1C_velocity
    from tkspace.forward_models.onedim.fm_utilities import parker2006AIF
    from tkspace.utilities import tidy
    from scipy.signal.windows import gaussian as gaussian
    from sigfig import round as sfround


    int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry()

    dx, dy, dz = res
    Lx, Ly, Lz, T = sysdim
    dxyz, dxz, dyz = dx*dy*dz, dx*dz, dy*dz
    
    fmax, umax, Dmax, psDmax, Fmax, Kvamax, Jmax = max_vals
    
    Nx, Ny, Nz = meas_voxels
    xc = np.linspace(dx/2,Lx-dx/2,Nx)
    
    if case == 1:
        
        v =(0.6*(0.4*np.sin(0.3*xc/10)**2+0.6*np.cos(0.15*xc/10)**2)+0.3)
    
        D = np.zeros(Nx+1)
        
        f = np.ones((Nx+1))*10 # mL/s/cm^2

        u = np.zeros_like(f)
        
        for i in range(0,len(f)):
            if f[i] < 0:
                if i == len(D)-1:
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
        dt = sfround(dt,1)    
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt)
        tmin = t/60
        delay = int(10/dt)
        Jl = parker2006AIF(tmin[0:-delay])
        Jl = np.append(np.zeros(delay),Jl)
        Jr = np.zeros(Nt)

    
        P = np.append(u,D)
        P = np.append(P,Jl)
        P = np.append(P,Jr)
        
        Pmax = np.tile(umax,u.size)
        Pmax = np.append(Pmax,np.tile(Dmax,D.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jl.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jr.size))
        
        fdir = 'case_dictionaries/case1_1d1c'

    if case == 3:
        
        v =(0.6*(0.5*np.sin(0.01*xc)**2+0.6*np.cos(0.02*xc)**2)+0.3)
    
        D = np.zeros(Nx+1)
        
        f = np.ones((Nx+1))*-6 # mL/s/cm^2

        u = np.zeros_like(f)
        
        for i in range(0,len(f)):
            if f[i] < 0:
                if i == len(D)-1:
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
        dt = sfround(dt,1)    
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt)
        tmin = t/60
        delay = int(15/dt)
        Jr = parker2006AIF(tmin[0:-delay])
        Jr = np.append(np.zeros(delay),Jr)
        Jl = np.zeros(Nt)

    
        P = np.append(u,D)
        P = np.append(P,Jl)
        P = np.append(P,Jr)
        
        Pmax = np.tile(umax,u.size)
        Pmax = np.append(Pmax,np.tile(Dmax,D.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jl.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jr.size))
        
        fdir = 'case_dictionaries/case3_1d1c'
        
    if case == 2:
        
        v =(0.6*(0.5*np.sin(0.02*xc)**2+0.6*np.cos(0.03*xc)**2)+0.3)
    
        D = np.zeros(Nx+1)
        
        f = np.ones((Nx+1))*5 # mL/s/cm^2

        u = np.zeros_like(f)
        
        for i in range(0,len(f)):
            if f[i] < 0:
                if i == len(D)-1:
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
        dt = sfround(dt,1)    
        Nt = int(T/dt)
        
        t=np.linspace(0,T,Nt)
        tmin = t/60
        delay = int(15/dt)
        Jl = parker2006AIF(tmin[0:-delay])
        Jl = np.append(np.zeros(delay),Jl)
        Jr = np.zeros(Nt)

    
        P = np.append(u,D)
        P = np.append(P,Jl)
        P = np.append(P,Jr)
        
        Pmax = np.tile(umax,u.size)
        Pmax = np.append(Pmax,np.tile(Dmax,D.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jl.size))
        Pmax = np.append(Pmax,np.tile(Jmax,Jr.size))
        
        fdir = 'case_dictionaries/case2_1d1c'
    np.savez_compressed(fdir, f=f[0], v=v, u = u, D = D, Jl = Jl, Jr = Jr, t =t, Nt = Nt, dt = dt, P = P, Pmax = Pmax)
    
    msg = '.npz output at {}'.format(fdir)
    
    return msg
