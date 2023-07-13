#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing functions to allocate the physical system geometries.
"""

def geometry():
    
    '''Defines set 1D geometry
    
    Parameters:
        
        - None
    
    Returns:
        
        - int_res (list):  
            - Grouped internal simulation resolution. [dx,dy,dz,dtmin].  
        - meas_res (list):  
            - Grouped measurement sampling resolution. [dx,dy,dz,dtmin].  
        - sysdim (list):  
            - System total dimensions. [Lx,Ly,Lz, T].  
        - voxels (list):  
            - Internal resolution voxel number in each axis. [Nx, Ny, Nz].  
        - meas_voxels (list):  
            - Measurment resolution voxel number in each axis.[Nx, Ny, Nz].  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
    
    '''
    
    Lx, Ly, Lz = 256, 0.8, 0.9 # Specfiy system length parameters (mm)
    dx, dy, dz = 0.1, 0.8, 0.9 # Specify forward model ground truth model spatial resolution (mm)
    Dx, Dy, Dz = 8, Ly, Lz # Specify Downsampling 'measurement' spatial resolution (mm)

    T   =  80# 0.0001 Sepcify temporal reso;utions for minimum cfl time, sampling time and total system evolution time
    
    max_vals = 14, 35, 4.5e-4, 4.5e-3, 0.07, 0.7, 0.8 * (Lx*Ly*Lz) * 1e-3#fmax3.05, umax, Dmax, psDmax, Fmax, Kvamax, Jamax, Jvmax
    
    Nx, Ny, Nz = int(Lx/dx), int(Ly/dy), int(Lz/dz)
    NX, NY, NZ = int(Lx/Dx), int(Ly/Dy), int(Lz/Dz)
    
    int_res     = [dx,dy,dz]
    meas_res    = [Dx,Dy,Dz]
    sysdim      = [Lx,Ly,Lz, T] 
    voxels      = [Nx, Ny, Nz] 
    meas_voxels = [NX, NY, NZ]

    return int_res, meas_res, sysdim, voxels, meas_voxels, max_vals

def multires_1d(dx,Lx):
    
    '''Allocates multiresolution if needed.
    
    Parameters:
        
        - dx (int OR numpy.ndarray inat64):  
            - Spatial resolution. Either for single resolution or multi resolution system.  
        - Lx (float64):  
            - Physical length of the system in x direction (cm).  
        
    Returns:
        
        - Nx (int OR numpy.ndarray inat64):  
            - Voxel number. Either for single resolution or multi resolution system.  
        - dx (int OR numpy.ndarray inat64):  
             - Spatial resolution. Either for single resolution or multi resolution system.  
        - xc_fine (numpy.ndarray float64):  
            - Central voxel values (cm) for single resolution or finest resolution of a multi resolution system.  
        - x_fine (numpy.ndarray float64):  
            - Interface voxel values (cm) for single resolution or finest resolution of a multi resolution system.  
    '''
    
    import numpy as np
    
    
    if type(dx) != int and type(dx) != float:
        Nx = (Lx/dx).astype(int)
        xc_fine = np.linspace(dx[-1]/2,Lx-dx[-1]/2,Nx[-1]) # finest central mesh to be used
        x_fine = np.linspace(0,Lx,Nx[-1]+1) # finest face centered mesh to be used
    else:
        Nx = int(Lx/dx)
        xc_fine = np.linspace(dx/2,Lx-dx/2,Nx)
        x_fine = np.linspace(0,Lx,Nx+1)
    
    return Nx, dx, xc_fine, x_fine
    
    
