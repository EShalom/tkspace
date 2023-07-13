#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:09:50 2021

@author: pyess
"""
def extract_last_line(filename):
    '''Method followed from https://openwritings.net/pg/python/python-read-last-line-file '''
    import os
    from numpy import array
    with open(filename, "rb") as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR) 
        Pstr = f.readline().decode()
    
    return array(Pstr[0][:-1].split(','),float)
    
def Plist_comp_1d2c_vel(path_to_output):
    
    import numpy as np
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d2c
    
    args = np.load('{}/args_in_opt.npz'.format(path_to_output))
    
    x = args['x_fine']
    xc = args['xc_fine']
    tsamp = args['tsamp']
    t = args['t']
    sysdim = args['sysdim']
    dx, dy, dz = args['meas_res']
    C_tiss = args['C_tiss']
    Ca = args['Ca']
    Cv = args['Cv']
    
    Nx = int(Lx/dx)
    Pinvnorm = extract_last_line('{}/Plist.txt'.format(path_to_output),1)
    Pinvmax = extract_last_line('{}/Pinvmax.txt'.format(path_to_output),1)
    Pinv = Pinvnorm * Pinvmax
    
    rec_vals = param_extract_vel_1d2c(Pinv,Nx)
    gt_vals = args['gt_vals']
    init = args['init_vals']
    
    dt = t[1]-t[0]
    dxyz = dx*dy*dz
    
    plot_1d2c_vel(path_to_output, rec_vals, gt_vals,init, C_tiss, Ca, Cv, sysdim,x,xc,tsamp,t,dt, dxyz)
    return

def param_comp_1d2c_vel(path_to_output):
    
    import numpy as np

    args = np.load('{}/args_in_opt.npz'.format(path_to_output), allow_pickle=True)
    opt = np.load('{}/FinalOptimisationValues.npz'.format(path_to_output), allow_pickle=True)
    
    rec_vals = opt['value_tuple']
    gt_vals = args['gt_vals']
    init = args['init_vals']
    
    ua_inv, uv_inv, Kva_inv, Da_inv, Dv_inv, Jpa_inv, Jpv_inv, Jna_inv, Jnv_inv = rec_vals
    ua, uv, Da, Dv, Kva, Jpa, Jpv, Jna, Jnv = gt_vals
    ua_init, uv_init, Kva_init, Da_init, Dv_init, Jpa_init, Jna_init, Jpv_init, Jnv_init = init
    
    x = args['x_fine']
    xc = args['xc_fine']
    tsamp = args['tsamp']
    t = args['t']
    sysdim = args['sysdim']
    dx, dy, dz = args['meas_res']
    C_tiss = args['C_tiss']
    Ca = args['Ca']
    Cv = args['Cv']
    
    dt = t[1]-t[0]
    dxyz = dx*dy*dz
    
    plot_1d2c_vel(path_to_output, rec_vals, gt_vals,init, C_tiss, Ca, Cv, sysdim,x,xc,tsamp,t,dt, dxyz)
    
    return


def plot_1d2c_vel(path_to_output, rec_vals, gt_vals,init, C_tiss, Ca, Cv,sysdim,x,xc,tsamp,t,dt, dxyz):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from TKfunctions.forward_models.onedim.velocity import onedim_twocomp_vel_diff
    import numpy as np
    
    Lx, Ly, Lz, T = sysdim                                                    
    ua_inv, uv_inv, Kva_inv, Da_inv, Dv_inv, Jpa_inv, Jpv_inv, Jna_inv, Jnv_inv = rec_vals
    ua, uv, Da, Dv, Kva, Jpa, Jpv, Jna, Jnv = gt_vals
    ua_init, uv_init, Kva_init, Da_init, Dv_init, Jpa_init, Jna_init, Jpv_init, Jnv_init = init
    
    plt.rcParams.update({'font.size':24})
    
    fig = plt.figure(constrained_layout=True,figsize=[20,15])
    gs = GridSpec(3,3,hspace=0.05,wspace=0.05, bottom=0.3,
                       left=0.02, right=0.95,figure=fig)# hspace=0.4,wspace=0.1
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    
    ax1.set(xlim=([0,Lx]),xlabel='x (cm)',ylabel='Kva \n (s$^{-1}$)')
    ax2.set(xlim=([0,Lx]),xlabel='x (cm)',ylabel='ua \n (cm s$^{-1}$)')
    ax3.set(xlim=([0,Lx]),xlabel='x (cm)',ylabel='uv \n (cm s$^{-1}$)')
    ax4.set(xlim=([0,T]),ylabel='Jpa (mM/s)',xlabel='t (s)')
    ax5.set(xlim=([0,T]),ylabel='Jna (mM/s)',xlabel='t (s)')
    ax6.set(xlim=([0,T]),xlabel='t (s)',ylabel='Jpv (mM/s)')
    ax7.set(xlim=([0,T]),xlabel='t (s)',ylabel='Jnv (mM/s)')
    ax8.set(xlim=([0,Lx]),xlabel='x (cm)',ylabel='Da \n (cm s$^{-2}$)')
    ax9.set(xlim=([0,Lx]),xlabel='x (cm)',ylabel='Dv \n (cm s$^{-2}$)')

    

    ax1.plot(xc,Kva,'k')
    ax1.plot(xc,Kva_inv,'*r')
    ax1.plot(xc,Kva_init,'*b')
        
    ax2.plot(x,ua,'k')
    ax2.plot(x,ua_inv,'*r')
    ax2.plot(x,ua_init,'*b')
    
    ax3.plot(x,uv,'k')
    ax3.plot(x,uv_inv,'*r')
    ax3.plot(x,uv_init,'*b')
    
    ax4.plot(t,Jpa,'k')
    ax4.plot(tsamp[:-1],Jpa_inv,'*r')
    
    ax5.plot(t,Jna,'k')
    ax5.plot(tsamp[:-1], Jna_inv,'*r')
    
    ax6.plot(t,Jnv,'k')
    ax6.plot(tsamp[:-1], Jnv_inv,'*r')
    
    ax7.plot(t,Jnv,'k')
    ax7.plot(tsamp[:-1], Jnv_inv,'*r')
    
    ax8.plot(x,Da,'k')
    ax8.plot(x,Da_inv,'*r')
    ax8.plot(x,Da_init,'*b')
    
    ax9.plot(x,Da,'k')
    ax9.plot(x,Da_inv,'*r')
    ax9.plot(x,Da_init,'*b')
    fig.savefig('{}/param_results_2.jpeg'.format(path_to_output))
    plt.close(fig)
    
    C_inv, Ca_inv, Cv_inv = onedim_twocomp_vel_diff(Lx, 1, tsamp, dt, ua_inv, uv_inv, Kva_inv, Da_inv, Dv_inv, np.append(Jpa_inv,Jpa_inv[-1]) / (dxyz * 1e-3), np.append(Jpv_inv,Jpa_inv[-1])/ (dxyz * 1e-3), np.append(Jna_inv,Jna_inv[-1])/ (dxyz * 1e-3), np.append(Jnv_inv,Jnv_inv[-1])/ (dxyz * 1e-3))

    plt.rcParams.update({'font.size':24})
    fig1 = plt.figure(constrained_layout=True,figsize=[15,20])
    gs = GridSpec(3,1,hspace=0.05,wspace=0.05, bottom=0.3,
                       left=0.02, right=0.95,figure=fig1)# hspace=0.4,wspace=0.1

    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[1, 0])
    ax3 = fig1.add_subplot(gs[2, 0])

    ax1.plot(xc,C_inv[:,0:-1:50])
    ax1.set_prop_cycle(None)
    ax1.plot(xc,C_tiss[:,0:-1:50],'--')
    ax1.set(xlabel='x (cm)',ylabel='Concentration (mmol/L)',title='Tissue Concentration')

    ax2.plot(xc,Ca_inv[:,0:-1:50])
    ax2.set_prop_cycle(None)
    ax2.plot(xc,Ca[:,0:-1:50],'--')
    ax2.legend(['10','20','30','40','50','60','70','80','90','100'],bbox_to_anchor=(1.3, 1.05),loc='upper right',fontsize=16,title='At time (s):')
    ax2.set(xlabel='x (cm)',ylabel='Concentration (mmol/L)',title='Arterial Tissue Concentration')

    ax3.plot(xc,Cv_inv[:,0:-1:50])
    ax3.set_prop_cycle(None)
    ax3.plot(xc,Cv[:,0:-1:50],'--')
    ax3.set(xlabel='x (cm)',ylabel='Concentration (mmol/L)',title='Venous Tissue Concentration')
    fig1.savefig('{}/concentration_results.tif'.format(path_to_output))
    plt.close(fig1)
    return