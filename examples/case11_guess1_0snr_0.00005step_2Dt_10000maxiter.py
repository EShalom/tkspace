#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:47:31 2022

@author: pyess
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:07:27 2021

@author: pyess
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import gmtime, strftime

from TKfunctions.utilities import mkdir_p as mkdir
from TKfunctions.forward_models.onedim.sampling import downsampler
from TKfunctions.forward_models.onedim.sampling import gaussian_noise as noise
from TKfunctions.forward_models.onedim.systeminfo import geometry_cm
from TKfunctions.forward_models.onedim.systeminfo import multires_1d as multires
from TKfunctions.forward_models.onedim.loadcase import load_P

from TKfunctions.inversion_methods.gradient_descent import gradient_descent_1d2c_flow_setup as inverse_method
from TKfunctions.inversion_methods.gradient_descent import simple_gradient_descent as method
from TKfunctions.forward_models.onedim.flow import onedim_twocomp_flow_diff as fmod
from TKfunctions.forward_models.onedim.fm_utilities import fmodsetup_1d2c as fmod_setup
from TKfunctions.forward_models.onedim.cases import case_1D2C_cm as gen_case
from TKfunctions.forward_models.onedim.loadcase import load_flow_1d2c as load_kinetics

from TKfunctions.forward_models.onedim.fm_utilities import convertTtoL
from TKfunctions.forward_models.onedim.flow import onedim_twocomp_flow_diff as fitmod
from TKfunctions.inversion_methods.cost_functions import rms_psp_cost as cost

dims        = 1 # what dimension to be used
pic         = 5 # 1 = normal local conc, 2= tissue conc
comp        = 2 # true compartment number
fitcomp     = 2   # number of compartments considered in the fit

guess_no    = 1 # guess type (1 - all optimsable params is 1 stack, 2 - multiple Pfix columns for multi stage fits)
case        = 11 # case number for system type (detailed in testsystems.py)
SNR         = 0 # SNR for run
Dt          = 2  # sampling time (s)
maxiter     = 10000
updatemethod= 'adam'
step        = 0.00005
fdir        = 'case11_guess1_0snr_0.00005step_2Dt_10000maxiter'#{}_flow_{}d{}c_guess{}_case{}_step{}_SNR{}_Dt{}_{}opt_maxiter{}'.format(strftime("%Y%m%d"),dims, comp, guess_no, case, step,SNR,Dt,updatemethod,maxiter)  #21112021_gtinterp/adams_2ogeval_step00005_tres_T80_1000iter_{}d{}c_pic{}_case{}_SNR{}_Dt{}_g{}_dx1'.format(dims,comp,pic,case,SNR,Dt,guess_no)
mkdir(fdir)
#%%
int_res, meas_res, sysdim, voxels, meas_voxels, max_vals = geometry_cm()
#gen_case(case,meas_res)
Lx, Ly, Lz, T = sysdim 
Nx_sim, Ny, Nz = voxels
dx_sim, dy, dz = int_res
dx, dy, dz = meas_res
dxyz = dx_sim * dy * dz 

fa, fv, Da, Dv, F, va, va_frac, vv, v, Jpa, Jpv, Jna, Jnv, dt, Nt, t = load_kinetics(dims,comp,case)
gt_vals = [fa[0], fv[0],fa, fv, F, Da, Dv, va, va_frac, vv, v, Jpa, Jpv, Jna, Jnv]

C_tiss, Ca, Cv = fmod(Lx, dx, t, dt, fa, fv, F, Da, Dv, va, vv, Jpa, Jpv, Jna, Jnv)
C_ds, tsamp = downsampler(C_tiss, T, dt, Dt, 0, 0)
Ca_ds, tsamp = downsampler(Ca, T, dt, Dt, 1, tsamp)
Cv_ds, tsamp = downsampler(Cv, T, dt, Dt, 1, tsamp)

Nt_samp = len(tsamp)
print("Parameters allocated \n")

# Signal to noise ratio ( 0 = noiseless) calculated as: noise stdev = mean(signal)/SNR

C_meas = noise(C_ds, SNR) # add guassian noise at scale of CNR

print("Downsample and noise addition complete \n")

# downsample known parameters to directly compare with inversion outputs
# allocate multi res options if using

Nx, dx, xc_fine, x_fine = multires(dx, Lx)
xc_sim = np.linspace(dx/2,Lx-dx/2,Nx) # finest central mesh to be used
x_sim = np.linspace(0,Lx,Nx+1) # finest face centered mesh to be used

print("Forwards model complete \n")

print("Starting Optimisation.... \n")

# create output folder with open files for cost function results and Param lists
Pnorm, Pmax, dt, Nt, t = load_P(dims, comp, case)


args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic]
#%%
np.savez_compressed('{}/args_in_opt'.format(fdir),x_fine=x_fine,xc_fine=xc_fine,C_meas=C_meas,Ca_ds=Ca_ds,Cv_ds=Cv_ds, tsamp=tsamp, C_tiss=C_tiss, t=t, fitmod=fitmod, Pnorm=Pnorm,Pmax=Pmax,int_res=int_res,meas_res=meas_res, dims=str(dims),comp=comp,case=case)

vguess = C_meas[:,-1]/max(C_meas[:,-1])

fa1_guess = [1.2,1.2,0.5,0.8,0.3]
fv1_guess = [-1.2,-0.3,-1,-0.8,-1.2]
F_guess = [0.065,0.055,0.045,0.035,0.025]
va_frac_guess = [0.4,0.45,0.5,0.55,0.6]

fa1_init = fa1_guess[guess_no-1]
fv1_init = fv1_guess[guess_no-1]
F_init = np.ones(Nx)*F_guess[guess_no-1]
Da_init=np.zeros(Nx+1)
Dv_init=np.zeros(Nx+1)
v=vguess
va_frac=np.ones(Nx)*va_frac_guess[guess_no-1]

start_time = int(np.ceil(24/Dt))#len(tsamp)
Jpa_init = C_meas[0,:start_time-1]#np.zeros(start_time-1)
Jna_init = C_meas[-1,:start_time-1]#np.zeros(start_time-1)
Jpv_init = np.zeros(start_time-1)
Jnv_init = np.zeros(start_time-1)
init_og = [fa1_init, fv1_init, F_init, Da_init, Dv_init,v, va_frac, Jpa_init, Jna_init, Jpv_init, Jnv_init]
init = [fa1_init, fv1_init, F_init, Da_init, Dv_init,v, va_frac, Jpa_init, Jna_init, Jpv_init, Jnv_init]
fix_vals = [1,1,0,1,1,1,0] #(f, F, D, v, va_frac,Ja, Jv)
lb = [-1,0,0,0.01,0.1,0]#(f, F, D, v, va_frac, J)
ub = [1,1,1,1,0.8,1]#(f, F, D, v, va_frac, J)
exit_change = [1e-4,1e-6,1e-9,1e-4,0]
max_vals = [1.4,0.15,4e-6,1,1,15] # # fmax, Fmax, Dmax, vmax, va_frac_max, Jmax

np.savez_compressed('{}/InitialOptimisationValues'.format(fdir), init_tuple=init, i=start_time)

#%%
for i in range(start_time, Nt_samp):
    args = [C_meas[:,0:i], fitmod, fmod_setup, meas_res, sysdim, tsamp[0:i], comp, pic, step,dx]

    value_tuple, Precon, fun, Pmax = inverse_method(method,updatemethod,init, fix_vals, lb, ub, exit_change, cost, maxiter, max_vals, fdir, args)
    fa1_inv, fv1_inv,fa_inv,fv_inv, F_inv, Da_inv, Dv_inv,va_inv,vaf_inv, vv_inv, v_inv, Jpa_inv, Jpv_inv, Jna_inv, Jnv_inv = value_tuple
    Jpa_init = np.append(Jpa_inv,Jpa_inv[-1])
    Jna_init = np.append(Jna_inv,Jna_inv[-1])
    Jpv_init = np.append(Jpv_inv,Jpv_inv[-1])
    Jnv_init = np.append(Jnv_inv,Jnv_inv[-1])
    init = [fa1_inv, fv1_inv, F_inv, Da_inv, Dv_inv,v_inv,vaf_inv, Jpa_init, Jna_init, Jpv_init, Jnv_init]
    print("Finshed fitting to {}s".format(tsamp[i]))
    print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    np.savez_compressed('{}/FinalOptimisationValues_{}s'.format(fdir,tsamp[i]), fun=fun, Precon=Precon,Pmax=Pmax, value_tuple=value_tuple, i=i)

fa1_inv, fv1_inv,fa_inv,fv_inv, F_inv, Da_inv, Dv_inv,va_inv,vaf_inv, vv_inv, v_inv, Jpa_inv, Jpv_inv, Jna_inv, Jnv_inv = value_tuple

print("finished! \n")
