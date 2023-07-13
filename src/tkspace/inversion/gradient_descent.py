#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing the functions associated with the gradient descent inversion method.
"""

def gradient_descent_1d1c_flow_setup(method, updatemethod, init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
    ''' Setup up parameter stacks for the gradient descent method algorithm, using local concentration picture with one compartment.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type. (f_init, D_init, v_init, Jp_init, Jn_init).  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    from sigfig import round as sfround    
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_flow_1d1c

    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp,comp, pic, step,dx = args
    f_init, D_init, v_init, Jp_init, Jn_init = init
    f_max, D_max, v_max, J_max = max_vals
    f_fix, D_fix, v_fix, J_fix = fix_vals
    f_ubnd, D_ubnd, v_ubnd, J_ubnd = ub_vals
    f_lbnd, D_lbnd, v_lbnd, J_lbnd = lb_vals
    f_econd, D_econd, v_econd, J_econd = econd_vals
    
    dx = meas_res[0]
    Lx = sysdim[0]
    Nx = int(Lx/dx)
    
    dtD = 0.1*(dx**2/D_max)
    dtf = 0.7*(dx/(f_max/0.3))
    dtsim = min(dtD, dtf)
    dtsim = sfround(dtsim, sigfigs=2)
    dtsim = 0.1
    
    f_norm  = f_init    / f_max
    D_norm  = D_init    / D_max
    v_norm = v_init / v_max
    Jp_norm = Jp_init   / J_max
    Jn_norm = Jn_init   / J_max
    
    f_econd  = f_econd    / f_max
    D_econd  = D_econd    / D_max
    v_econd = v_econd /v_max
    J_econd = J_econd   / J_max
    
    f_max = extend(f_init, f_max)
    D_max = extend(D_init,D_max)
    v_max = extend(v_init, v_max)
    Jp_max = extend(Jp_init, J_max)
    Jn_max = extend(Jn_init, J_max)
    
    f_fix = extend(f_init, f_fix)
    D_fix = extend(D_init,D_fix)
    v_fix = extend(v_init, v_fix)
    Jp_fix = extend(Jp_init, J_fix)
    Jn_fix = extend(Jn_init, J_fix)
    
    f_ubnd = extend(f_init, f_ubnd)
    D_ubnd = extend(D_init,D_ubnd)
    v_ubnd = extend(v_init, v_ubnd)
    Jp_ubnd = extend(Jp_init, J_ubnd)
    Jn_ubnd = extend(Jn_init, J_ubnd)
    
    f_lbnd = extend(f_init, f_lbnd)
    D_lbnd = extend(D_init,D_lbnd)
    v_lbnd = extend(v_init, v_lbnd)
    Jp_lbnd = extend(Jp_init, J_lbnd)
    Jn_lbnd = extend(Jn_init, J_lbnd)
    
    f_econd = extend(f_init, f_econd)
    D_econd = extend(D_init,D_econd)
    v_econd = extend(v_init, v_econd)
    J_econd = extend(Jp_init, J_econd)
    
    Pinitial = Pstack( (f_norm, D_norm, v_norm, Jp_norm, Jn_norm) )
    Pmax = Pstack( (f_max, D_max, v_max, Jp_max, Jn_max) )
    Pfix = Pstack( (f_fix, D_fix, v_fix, Jp_fix, Jn_fix) )
    Pecond = Pstack((f_econd, D_econd, v_econd, J_econd, J_econd))
    
    ub = Pstack( (f_ubnd, D_ubnd, v_ubnd, Jp_ubnd, Jn_ubnd) )
    lb = Pstack( (f_lbnd, D_lbnd, v_lbnd, Jp_lbnd, Jn_lbnd) )
    
    log = open_log(fdir, Pinitial, Pmax)
    
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic, step, dx]
    Pfinal, cost_value, Pmax = method(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args,updatemethod)
    output = param_extract_flow_1d1c(Pfinal*Pmax, Nx)
    
    return output, Pfinal, cost_value, Pmax

def gradient_descent_1d2c_flow_setup(method,updatemethod, init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
        
    ''' Setup up parameter stacks for the gradient descent method algorithm, using local concentration picture with 2 compartments.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type.  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    
    from sigfig import round as sfround    
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_flow_1d2c

    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic, step, dx = args
    fa_init, fv_init, F_init, Da_init, Dv_init,v_init, vaf_init, Jpa_init, Jna_init, Jpv_init, Jnv_init = init
    f_max, F_max,D_max, v_max, vaf_max, J_max = max_vals
    f_fix, F_fix, D_fix, v_fix, vaf_fix, Ja_fix, Jv_fix = fix_vals
    f_ubnd, D_ubnd, F_ubnd, v_ubnd, vaf_ubnd,J_ubnd = ub_vals
    f_lbnd, D_lbnd, F_lbnd,v_lbnd, vaf_lbnd, J_lbnd = lb_vals
    f_econd, F_econd, D_econd, v_econd,  J_econd = econd_vals
    
    dx = meas_res[0]
    Lx = sysdim[0]
    Nx = int(Lx/dx)
    
    dtD = 0.1*(dx**2/D_max)
    dtf = 0.7*(dx/(f_max/0.1))
    dtsim = min(dtD, dtf)
    dtsim = sfround(dtsim, sigfigs=1)
    
    fa_norm  = fa_init    / f_max
    fv_norm  = fv_init    / f_max
    Da_norm  = Da_init    / D_max
    Dv_norm  = Dv_init    / D_max
    F_norm = F_init / F_max
    v_norm = v_init / v_max
    vaf_norm = vaf_init / v_max
    Jpa_norm = Jpa_init   / J_max
    Jna_norm = Jna_init   / J_max
    Jpv_norm = Jpv_init   / J_max
    Jnv_norm = Jnv_init   / J_max
    
    f_econd  = f_econd    / f_max
    D_econd  = D_econd    / D_max
    F_econd = F_econd / F_max
    v_econd = v_econd /v_max
    vaf_econd = v_econd /vaf_max
    J_econd = J_econd   / J_max
    
    f_max = extend(fa_init, f_max)
    D_max = extend(Da_init,D_max)
    v_max = extend(v_init, v_max)
    vaf_max = extend(vaf_init, vaf_max)
    F_max = extend(F_init, F_max)
    J_max = extend(Jpa_init, J_max)
    
    f_fix = extend(fa_init, f_fix)
    D_fix = extend(Da_init,D_fix)
    F_fix = extend(F_init, F_fix)
    v_fix = extend(v_init, v_fix)
    vaf_fix = extend(vaf_init, vaf_fix)
    Ja_fix = extend(Jpa_init, Ja_fix)
    Jv_fix = extend(Jpv_init, Jv_fix)
    
    f_ubnd = extend(fa_init, f_ubnd)
    D_ubnd = extend(Da_init,D_ubnd)
    v_ubnd = extend(v_init, v_ubnd)
    vaf_ubnd = extend(vaf_init, vaf_ubnd)
    F_ubnd = extend(F_init, F_ubnd)
    J_ubnd = extend(Jpa_init, J_ubnd)
    
    f_lbnd = extend(fa_init, f_lbnd)
    D_lbnd = extend(Da_init,D_lbnd)
    v_lbnd = extend(v_init, v_lbnd)
    vaf_lbnd = extend(vaf_init, vaf_lbnd)
    F_lbnd = extend(F_init, F_lbnd)
    J_lbnd = extend(Jpa_init, J_lbnd)
    
    f_econd = extend(fa_init, f_econd)
    D_econd = extend(Da_init,D_econd)
    vaf_econd = extend(vaf_init, vaf_econd)
    v_econd = extend(v_init, v_econd)
    F_econd = extend(F_init, F_econd)
    J_econd = extend(Jpa_init, J_econd)
    
    Pinitial = Pstack( (fa_norm, fv_norm, F_norm, Da_norm, Dv_norm,vaf_norm, v_norm, Jpa_norm, Jpv_norm, Jna_norm, Jnv_norm) )
    Pmax = Pstack( (f_max, f_max, F_max, D_max, D_max,vaf_max, v_max, J_max, J_max, J_max, J_max) )
    Pfix = Pstack( (f_fix, f_fix, F_fix, D_fix, D_fix,vaf_fix, v_fix, Ja_fix, Jv_fix, Ja_fix, Jv_fix ) )
    Pecond = Pstack( (f_econd, f_econd, F_econd, D_econd, D_econd,vaf_econd, v_econd, J_econd, J_econd, J_econd, J_econd ) )

    ub = Pstack( (f_ubnd, f_ubnd, F_ubnd, D_ubnd, D_ubnd, vaf_ubnd, v_ubnd, J_ubnd, J_ubnd, J_ubnd, J_ubnd) )
    lb = Pstack( (f_lbnd, f_lbnd, F_lbnd,D_lbnd, D_lbnd, vaf_lbnd, v_lbnd, J_lbnd,J_lbnd,J_lbnd,J_lbnd) )
    
    log = open_log(fdir, Pinitial, Pmax)
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic, step, dx]
      
    Pfinal, cost_value, Pmax = method(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args,updatemethod)
    output = param_extract_flow_1d2c(Pfinal*Pmax, Nx, dx)
    
    return output, Pfinal, cost_value, Pmax


def gradient_descent_1d1c_vel_setup(method,init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
        
    ''' Setup up parameter stacks for the gradient descent method algorithm, using tissue concentration picture with one compartment.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type. (u_init, D_init, v_init, Jp_init, Jn_init).  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    
    from sigfig import round as sfround 
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d1c

    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic,step = args
    u_init, D_init, Jp_init, Jn_init = init
    u_max, D_max, J_max = max_vals
    u_fix, D_fix, J_fix = fix_vals
    u_ubnd, D_ubnd, J_ubnd = ub_vals
    u_lbnd, D_lbnd, J_lbnd = lb_vals
    u_econd, D_econd,  J_econd = econd_vals

    dx = meas_res[0]
    Lx = sysdim[0]
    Nx = int(Lx/dx)
    
    dtD = 0.1*(dx**2/D_max)
    dtu = 0.7*(dx/u_max)
    dtsim = min(dtD, dtu)
    dtsim = sfround(dtsim, sigfigs=2)
    
    u_norm  = u_init    / u_max
    D_norm  = D_init    / D_max
    Jp_norm = Jp_init   / J_max
    Jn_norm = Jn_init   / J_max
    
    u_econd  = u_econd    / u_max
    D_econd  = D_econd    / D_max
    J_econd = J_econd   / J_max
    
    u_max = extend(u_init, u_max)
    D_max = extend(D_init,D_max)
    Jp_max = extend(Jp_init, J_max)
    Jn_max = extend(Jn_init, J_max)
    
    u_fix = extend(u_init, u_fix)
    D_fix = extend(D_init,D_fix)
    Jp_fix = extend(Jp_init, J_fix)
    Jn_fix = extend(Jn_init, J_fix)
    
    u_ubnd = extend(u_init, u_ubnd)
    D_ubnd = extend(D_init,D_ubnd)
    Jp_ubnd = extend(Jp_init, J_ubnd)
    Jn_ubnd = extend(Jn_init, J_ubnd)
    
    u_lbnd = extend(u_init, u_lbnd)
    D_lbnd = extend(D_init,D_lbnd)
    Jp_lbnd = extend(Jp_init, J_lbnd)
    Jn_lbnd = extend(Jn_init, J_lbnd)

    u_econd = extend(u_init, u_econd)
    D_econd = extend(D_init,D_econd)
    J_econd = extend(Jp_init, J_econd)
    
    Pinitial = Pstack( (u_norm, D_norm, Jp_norm, Jn_norm) )
    Pmax = Pstack( (u_max, D_max, Jp_max, Jn_max) )
    Pfix = Pstack( (u_fix, D_fix, Jp_fix, Jn_fix) )
    Pecond = Pstack( (u_econd, D_econd, J_econd, J_econd))

    ub = Pstack( (u_ubnd, D_ubnd, Jp_ubnd, Jn_ubnd) )
    lb = Pstack( (u_lbnd, D_lbnd, Jp_lbnd, Jn_lbnd) )
    
    log = open_log(fdir, Pinitial, Pmax)
    
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic,step]
    
    Pfinal, cost_value, Pmax = method(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args)
    
    output = param_extract_vel_1d1c(Pfinal*Pmax, Nx)

    
    return output, Pfinal, cost_value, Pmax
    
def gradient_descent_1d2c_vel_setup(method,init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
        
    ''' Setup up parameter stacks for the gradient descent method algorithm, using tissue concentration picture with two compartments.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type. (u_init, D_init, v_init, Jp_init, Jn_init).  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    
    from sigfig import round as sfround
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d2c
    
    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic, step,dx = args
    ua_init, uv_init, Kva_init, Da_init, Dv_init, Jpa_init, Jna_init, Jpv_init, Jnv_init = init
    u_max, Kva_max,D_max,  J_max = max_vals
    u_fix, Kva_fix, D_fix,  Ja_fix, Jv_fix = fix_vals
    u_econd, Kva_econd, D_econd,  J_econd = econd_vals
    u_ubnd, D_ubnd, Kva_ubnd, J_ubnd = ub_vals
    u_lbnd, D_lbnd, Kva_lbnd, J_lbnd = lb_vals
    
    dx = meas_res[0]
    Lx = sysdim[0]
    Nx = int(Lx/dx)
    
    dtD = 0.1*(dx**2/D_max)
    dtu = 0.7*(dx/u_max)
    dtsim = min(dtD, dtu)
    dtsim = sfround(dtsim, sigfigs=1)
    
    ua_norm  = ua_init    / u_max
    uv_norm  = uv_init    / u_max
    Da_norm  = Da_init    / D_max
    Dv_norm  = Dv_init    / D_max
    Kva_norm = Kva_init / Kva_max
    Jpa_norm = Jpa_init   / J_max
    Jna_norm = Jna_init   / J_max
    Jpv_norm = Jpv_init   / J_max
    Jnv_norm = Jnv_init   / J_max
    
    u_econd  = u_econd    / u_max
    D_econd  = D_econd    / D_max
    Kva_econd = Kva_econd / Kva_max
    J_econd = J_econd   / J_max
    
    u_max = extend(ua_init, u_max)
    D_max = extend(Da_init,D_max)
    Kva_max = extend(Kva_init, Kva_max)
    J_max = extend(Jpa_init, J_max)
    
    u_fix = extend(ua_init, u_fix)
    D_fix = extend(Da_init,D_fix)
    Kva_fix = extend(Kva_init, Kva_fix)
    Ja_fix = extend(Jpa_init, Ja_fix)
    Jv_fix = extend(Jpv_init, Jv_fix)
    
    u_ubnd = extend(ua_init, u_ubnd)
    D_ubnd = extend(Da_init,D_ubnd)
    Kva_ubnd = extend(Kva_init, Kva_ubnd)
    J_ubnd = extend(Jpa_init, J_ubnd)
    
    u_lbnd = extend(ua_init, u_lbnd)
    D_lbnd = extend(Da_init,D_lbnd)
    Kva_lbnd = extend(Kva_init, Kva_lbnd)
    J_lbnd = extend(Jpa_init, J_lbnd)
    
    u_econd = extend(ua_init, u_econd)
    D_econd = extend(Da_init,D_econd)
    Kva_econd = extend(Kva_init, Kva_econd)
    J_econd = extend(Jpa_init, J_econd)
    
    Pinitial = Pstack( (ua_norm, uv_norm, Kva_norm, Da_norm, Dv_norm, Jpa_norm, Jpv_norm, Jna_norm, Jnv_norm) )
    Pmax = Pstack( (u_max, u_max, Kva_max, D_max, D_max, J_max, J_max, J_max, J_max) )
    Pfix = Pstack( (u_fix, u_fix, Kva_fix, D_fix, D_fix, Ja_fix, Jv_fix, Ja_fix, Jv_fix ) )
    Pecond = Pstack( (u_econd, u_econd, Kva_econd, D_econd, D_econd, J_econd, J_econd, J_econd, J_econd))
    
    ub = Pstack( (u_ubnd, u_ubnd, Kva_ubnd, D_ubnd, D_ubnd, J_ubnd, J_ubnd, J_ubnd, J_ubnd) )
    lb = Pstack( (u_lbnd, u_lbnd, Kva_lbnd,D_lbnd, D_lbnd, J_lbnd,J_lbnd,J_lbnd,J_lbnd) )
    
    log = open_log(fdir, Pinitial, Pmax)
    
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic, step, dx]
    
    Pfinal, cost_value, Pmax = method(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args)
    
    output = param_extract_vel_1d2c(Pfinal*Pmax, Nx)
    
    return output, Pfinal, cost_value, Pmax

def gradient_descent_1d2c_vel_msres_setup(method,init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
        
    ''' Setup up parameter stacks for the gradient descent method algorithm, using tissue concentration picture with two compartments.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type. (u_init, D_init, v_init, Jp_init, Jn_init).  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    
    from sigfig import round as sfround
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d2c_msres
    from numpy import ceil
    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic, step,dx = args
    ua_init, uv_init, Kva_init, Da_init, Dv_init, Ja_init, Jpa_f_init, Jv_init, Jpv_f_init = init
    u_max, Kva_max, D_max,  J_max, Jf_max = max_vals
    u_fix, Kva_fix, D_fix,  Ja_fix, Jv_fix = fix_vals
    u_econd, Kva_econd, D_econd,  J_econd, Jf_econd = econd_vals
    u_ubnd, D_ubnd, Kva_ubnd, J_ubnd, Jf_ubnd = ub_vals
    u_lbnd, D_lbnd, Kva_lbnd, J_lbnd, Jf_lbnd = lb_vals
    
    Lx = sysdim[0]
    Nx = (Lx/dx).astype(int)
    dtsim = 0.7*(dx/u_max)
    dtsim[dtsim>=0.2]=0.2
    dtsim[dtsim<0.2]=0.02
    
    ua_norm  = ua_init    / u_max
    uv_norm  = uv_init    / u_max
    Da_norm  = Da_init    / D_max
    Dv_norm  = Dv_init    / D_max
    Kva_norm = Kva_init / Kva_max
    Jpa_f_norm = Jpa_f_init   / Jf_max
    Ja_norm = Ja_init   / J_max
    Jpv_f_norm = Jpv_f_init   / Jf_max
    Jv_norm = Jv_init   / J_max
    
    u_econd  = u_econd    / u_max
    D_econd  = D_econd    / D_max
    Kva_econd = Kva_econd / Kva_max
    J_econd = J_econd   / J_max
    Jf_econd = Jf_econd   / Jf_max
    
    u_max = extend(ua_init, u_max)
    D_max = extend(Da_init,D_max)
    Kva_max = extend(Kva_init, Kva_max)
    J_max = extend(Ja_init, J_max)
    Jf_max = extend(Jpa_f_init, Jf_max)
    
    u_fix = extend(ua_init, u_fix)
    D_fix = extend(Da_init,D_fix)
    Kva_fix = extend(Kva_init, Kva_fix)
    Ja_fix = extend(Ja_init, Ja_fix)
    Jv_fix = extend(Jv_init, Jv_fix)
    
    u_ubnd = extend(ua_init, u_ubnd)
    D_ubnd = extend(Da_init,D_ubnd)
    Kva_ubnd = extend(Kva_init, Kva_ubnd)
    J_ubnd = extend(Ja_init, J_ubnd)
    Jf_ubnd = extend(Jpa_f_init, Jf_ubnd)
    
    u_lbnd = extend(ua_init, u_lbnd)
    D_lbnd = extend(Da_init,D_lbnd)
    Kva_lbnd = extend(Kva_init, Kva_lbnd)
    J_lbnd = extend(Ja_init, J_lbnd)
    Jf_lbnd = extend(Jpa_f_init, Jf_lbnd)

    u_econd = extend(ua_init, u_econd)
    D_econd = extend(Da_init,D_econd)
    Kva_econd = extend(Kva_init, Kva_econd)
    J_econd = extend(Ja_init, J_econd)
    Jf_econd = extend(Jpa_f_init, Jf_econd)
    
    Pinitial = Pstack( (ua_norm, uv_norm, Kva_norm, Da_norm, Dv_norm, Ja_norm, Jpa_f_norm, Jv_norm, Jpv_f_norm) )
    Pmax = Pstack( (u_max, u_max, Kva_max, D_max, D_max, J_max, Jf_max, J_max, Jf_max) )
    Pfix = Pstack( (u_fix, u_fix, Kva_fix, D_fix, D_fix, Ja_fix, Ja_fix, Jv_fix, Jv_fix ) )
    Pecond = Pstack( (u_econd, u_econd, Kva_econd, D_econd, D_econd, J_econd, Jf_econd, J_econd, Jf_econd))
    
    ub = Pstack( (u_ubnd, u_ubnd, Kva_ubnd, D_ubnd, D_ubnd, J_ubnd, Jf_ubnd, J_ubnd, Jf_ubnd) )
    lb = Pstack( (u_lbnd, u_lbnd, Kva_lbnd,D_lbnd, D_lbnd, J_lbnd,Jf_lbnd,J_lbnd,Jf_lbnd) )
    
    log = open_log(fdir, Pinitial, Pmax)
    
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic, step,dx]
    
    Pfinal, cost_value, Pmax = method(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log,fdir, args)
    
    output = param_extract_vel_1d2c_msres(Pfinal*Pmax, Nx[-1])
    
    return output, Pfinal, cost_value, Pmax

def gradient_descent_1d2c_vel_globalopt(method,init, fix_vals, lb_vals, ub_vals, econd_vals, costfn, Niter, max_vals, fdir, args):
    
        
    ''' Setup up parameter stacks for the gradient descent method algorithm, using tissue concentration picture with two compartments.
    
    Parameters:
        
        - method (function):  
            - Calls either direct simple gradient descent or multiresolution setup.  
        - init (tuple):  
            - Initial values for each parameter type. (u_init, D_init, v_init, Jp_init, Jn_init).  
        - fix_vals (tuple): 
            - Index to show if free parameter in optimiser for each parameter type. 1 = free, 0 = set.  
        - lb_vals (tuple):  
            - Lower bound for each parameter type.  
        - ub_vals (tuple):  
            - Upper bound for each parameter type.  
        - econd_vals (tuple):  
            - exit condition change for each parameter type.  
        - costfn (function):  
            - Cost function to calculate quantity to optimise.  
        - Niter (int):  
            - Maximum number of iterations allowed.  
        - max_vals (tuple):  
            - Physiological maximum values for each parameter type.  
        - fdir (str):  
            - Output folder pathname.  
        - args (tuple):  
            - Arguement needed to run the forwards model within the cost function.  
            
    Returns:
        
        - Pfinal (numpy.ndarray):  
            - Final normalised parameter values after optimisation.  
        - cost_value (float64):  
            - Value of cost function at Pfinal.  
        - Pmax (numpy.ndarray):  
            - Maximum parameter values. Same size as Pfinal to rescale into physical quantities.  
        
    '''
    
    from sigfig import round as sfround
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d2c
    from time import perf_counter

    C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, pic = args
    ua_init, uv_init, Kva_init, Da_init, Dv_init, Jpa_init, Jna_init, Jpv_init, Jnv_init = init
    u_max, Kva_max,D_max,  J_max = max_vals
    u_fix, Kva_fix, D_fix,  Ja_fix, Jv_fix = fix_vals
    u_econd, Kva_econd, D_econd,  J_econd = econd_vals
    u_ubnd, D_ubnd, Kva_ubnd, J_ubnd = ub_vals
    u_lbnd, D_lbnd, Kva_lbnd, J_lbnd = lb_vals
    
    dx = meas_res[0]
    Lx = sysdim[0]
    Nx = int(Lx/dx)
    
    dtD = 0.1*(dx**2/D_max)
    dtu = 0.7*(dx/u_max)
    dtsim = min(dtD, dtu)
    dtsim = sfround(dtsim, sigfigs=1)
    
    ua_norm  = ua_init    / u_max
    uv_norm  = uv_init    / u_max
    Da_norm  = Da_init    / D_max
    Dv_norm  = Dv_init    / D_max
    Kva_norm = Kva_init / Kva_max
    Jpa_norm = Jpa_init   / J_max
    Jna_norm = Jna_init   / J_max
    Jpv_norm = Jpv_init   / J_max
    Jnv_norm = Jnv_init   / J_max
    
    u_econd  = u_econd    / u_max
    D_econd  = D_econd    / D_max
    Kva_econd = Kva_econd / Kva_max
    J_econd = J_econd   / J_max
    
    u_max = extend(ua_init, u_max)
    D_max = extend(Da_init,D_max)
    Kva_max = extend(Kva_init, Kva_max)
    J_max = extend(Jpa_init, J_max)
    
    u_fix = extend(ua_init, u_fix)
    D_fix = extend(Da_init,D_fix)
    Kva_fix = extend(Kva_init, Kva_fix)
    Ja_fix = extend(Jpa_init, Ja_fix)
    Jv_fix = extend(Jpv_init, Jv_fix)
    
    u_bnds = [[u_lbnd,u_ubnd]]*len(ua_init)
    D_bnds = [[0.0,0.000000000001]]*len(Da_init)
    Kva_bnds = [[Kva_lbnd,Kva_ubnd]]*len(Kva_init)
    J_bnds = [[J_lbnd,J_ubnd]]*len(Jpa_init)
    Jv_bnds = [[0.0,0.000000000001]]*len(Jpa_init)
    
    u_econd = extend(ua_init, u_econd)
    D_econd = extend(Da_init,D_econd)
    Kva_econd = extend(Kva_init, Kva_econd)
    J_econd = extend(Jpa_init, J_econd)
    
    Pinitial = Pstack( (ua_norm, uv_norm, Kva_norm, Da_norm, Dv_norm, Jpa_norm, Jpv_norm, Jna_norm, Jnv_norm) )
    Pmax = Pstack( (u_max, u_max, Kva_max, D_max, D_max, J_max, J_max, J_max, J_max) )
    
    bnds = u_bnds + u_bnds + Kva_bnds + D_bnds + D_bnds + J_bnds + Jv_bnds + J_bnds + Jv_bnds
    
    log = open_log(fdir, Pinitial, Pmax)
    costtxt, Plisttxt = log
    timestart = perf_counter()
    with open(costtxt,'w') as f:
        f.write(str(0)+'\t'+str(timestart)+'\t'+str(0))
        
    from numpy import savetxt,savez_compressed

    def save_callback(x):
        costtxt, Plisttxt = log
        savez_compressed('Outputs/lbfgsb_1d2c_pic2_case1_SNR0_Dt2/FinalOptimisationValues'.format(fdir), Precon=x)
        with open(Plisttxt,'ab') as file:
            file.write(b'\n')
            savetxt(file, x, fmt='%1.5f', delimiter=',',newline=',')
        return
    
    args = [C_meas, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic,Pmax]
    #opt_res = method(costfn, Pinitial, niter=25, T=1.0, stepsize=0.01, minimizer_kwargs={"method":"L-BFGS-B","bounds":bnds,"args":args}, take_step=None, accept_test=None, callback=save_callback, interval=50, disp=True, niter_success=None, seed=None)
    #opt_res = method(costfn, bnds, args=[args], maxiter=100, local_search_options={'method':'L-BFGS-B'}, initial_temp=5230.0, restart_temp_ratio=2e-05, visit=2.62, accept=- 5.0, maxfun=10000000.0, seed=None, no_local_search=False, callback=save_callback, x0=Pinitial)
    opt_res = method(costfn, Pinitial, args=args, method='L-BFGS-B', jac=None, bounds=bnds, tol=None, callback=save_callback, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-14, 'gtol': 1e-07, 'eps': 1e-10, 'maxfun': 15000, 'maxiter': 100, 'iprint': 100, 'maxls': 20, 'finite_diff_rel_step': None})
    
    Pfinal = opt_res.x
    cost_value = opt_res.fun
    print('status: {}, message: {} \n'.format(opt_res.status,opt_res.message))
    output = param_extract_vel_1d2c(Pfinal*Pmax, Nx)
    
    return output, Pfinal, cost_value, Pmax

def simple_gradient_descent(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args,updatemethod):
    
    ''' Simple gradient descent algorithm using first order gradient evaluation with either an adams update or backtracking type line search.
    
    Parameters:
        
        - Pinitial (numpy.ndarray float64):  
            - Initial parameter set for optimisation.  
        - Pfix (numpy.ndarray float64):  
            - Same length as Pinitial. Shows if parameter will be optimised in algorithm. Fixed = 0, Free = 1.  
        - Pecond (numpy.ndarray float64):  
            - Same length as Pinitial. Shows exit condition change for each parameter type.  
        - costfn (function):  
            - Python or other callable function with appropriate wrapper for calculating single valued cost function output.  
        - Niter (int):  
            - Maximum iterations for algorithm.  
        - ub (numpy.ndarray float64):  
            - Upper bounds for every parameter in Pinitial.  
        - lb (numpy.ndarray float64):  
            - Lower bounds for every parameter in Pinitial.  
        - log (list):  
            - List of log files to write out to contains: costtxt (cost finction readout), and Plisttxt (Parameters after every iteration).  
        - args (Tuple):  
            - Collection of extra arguements used by the cost function.  
        
    Returns:
        
        - Precon (numpy.ndarray float64):  
            - Final optimisation result parameter set.  
        - current (float64):  
            - Final cost function value using Precon.  
    '''
    import numpy as np
    from time import perf_counter
    from copy import deepcopy
    
    costtxt, Plisttxt = log
    
    timestart = perf_counter()
    cost_track = np.zeros(4)
    gtol = 0
    
    current = costfn(Pinitial,Pmax,args)
    cost_track[-1]=current
    
    timenow = perf_counter()-timestart
    
    with open(costtxt,'w') as f:
        f.write(str(current)+'\t'+str(timenow)+'\t'+str(current))
    
    Precon = deepcopy(Pinitial)
    mt =0
    vt =0
    eps = 1e-8
    b1 =0.9
    b2=0.9
    step = args[-2]
    
    for j in range(0,Niter):
        
        gradvec = gradeval_1stoder(Precon, Pmax, Pfix, ub, lb, step, costfn, current, args)
      
        if updatemethod == 'adam':
            Pupdate , vt, mt = adam_update(Precon,gradvec,step,0.9,mt,vt,eps,b1,b2,ub,lb)
        if updatemethod == 'linesearch':
            scaled_step = step_scaling(gradvec, Pfix)
            Pupdate = backtracking_linesearch(Precon, Pmax, 0.5, scaled_step, 0.9, ub, lb, costfn, current, args)
        updatecost = costfn(Pupdate, Pmax, args)
        write_log(costtxt, Plisttxt, Precon, current, updatecost, timestart)
        
        Pchange = abs(Precon - Pupdate)
        
        cost_track, gtol, exit_cond = check_exitcond(cost_track, gtol, Pchange, Pecond, Pfix, updatecost, current)
        
        if exit_cond != None:
            print(exit_cond)
            print('Iter reached: {}'.format(j+1))
            return Precon, current, Pmax
        
        current = updatecost
        Precon = deepcopy(Pupdate)
        
    print('Iter reached: {}'.format(j+1))
    return Precon, current, Pmax

def simulated_annealing(Pinitial, Pmax, Pfix,Pecond, costfn, Niter,ub,lb,log, args):
    """
    Created on Fri Jul 16 15:16:46 2021
    Based on example from the simulated annealing tutorial
    by Son Duy Dao found at learnwithpanda.com
    
    Edited to run with current code setup and as function rather than isolated script
    """
    import random
    import math
    import numpy as np
    from time import perf_counter

    timestart = perf_counter()

    costtxt, Plisttxt = log
    initial_temp = 1500
    cooling = 0.95
    
    current_sol = Pinitial
    best_sol = Pinitial

    n = 1
    best_fit = costfn(best_sol,Pmax,args)
    current_temp = initial_temp
    no_attempts = 100
    
    for i in range(Niter):
        print(i)
        old_fit = costfn(best_sol,1,args)
        best_sol, best_fit, Pmax = simple_gradient_descent(Pinitial, Pmax, Pfix, Pecond, costfn, 200, ub, lb, log, args)

        #print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        for j in range(no_attempts):
            current_sol = best_sol + (0.01*(np.random.uniform(-1,1,len(Pfix)))*Pfix)
            current_sol[current_sol>ub]=ub[current_sol>ub]
            current_sol[current_sol<lb]=lb[current_sol<lb]
            
            current_fit = costfn(current_sol,Pmax,args)
            E = abs(current_fit - best_fit)
            if i ==0 or j ==0:
                EA = E
            
            if current_fit > best_fit:
                p = math.exp(-E/(EA*current_temp))
                
                if random.random()<p:
                    accept = True
                else:
                    accept = False
                    
            else:
                accept = True
            
            if accept == True:
                best_sol = current_sol
                best_fit = costfn(best_sol,Pmax,args)
                n = n + 1
                EA = (EA * (n-1) + E)/n
        #print('interaction: {}, best_sol: {}, best_fit {}'.format(i,best_sol,best_fit))
        current_temp = current_temp*cooling
        
        write_log(costtxt, Plisttxt, best_sol, old_fit, best_fit, timestart)

        if best_fit < 1e-9:
            break
        if old_fit==best_fit:
            break
        
    return best_sol, best_fit,Pmax

def adam_update(Precon,gradvec,eta,shrink_factor, mt,vt,eps,b1,b2,ub,lb):
    from numpy import power, sqrt, heaviside
    from numpy import any as npany
    mt = (1-b1)*gradvec + b1*mt
    vt= (1-b2)*power(gradvec,2) + b2*vt
    
    mt_ub = mt/(1-b1)
    vt_ub = vt/(1-b2)
    
    update_factor = (eta * mt_ub)/(sqrt(vt_ub)+eps)
    Pupdate = Precon - update_factor
    
    Pupdate[Pupdate>ub]=ub[Pupdate>ub]
    Pupdate[Pupdate<lb]=lb[Pupdate<lb]
    #aob = 0
    zcross = 0 
# =============================================================================
#     while len(Pupdate[Pupdate > ub]) != 0 or len(Pupdate[Pupdate < lb]) != 0:
#         aob += 1
#         print('out of bounds # {}'.format(aob))
#         update_factor *= shrink_factor
#         Pupdate = Precon - update_factor
#     
# =============================================================================
    rescale_factors = []
    
    if npany((Pupdate*Precon) < 0):
        for i in range(0,len(Pupdate)):
            if heaviside(Pupdate[i] * Precon[i],1)==0 :
                zcross += 1
                rescale_factors += [abs(Precon[i])/abs(Precon[i]-Pupdate[i])]
        Pupdate = Precon - min(rescale_factors)*update_factor      
        return Pupdate, vt, mt
    else:
        return Pupdate, vt, mt
        
    
def gradeval_1stoder(Precon, Pmax, Pfix, ub, lb, eval_step, costfn, current, args):
    
    ''' Performs a 1st order gradient evalution of the effect of each parameter on the value of the cost function.
    
    Parameters:
        
        - Precon (numpy.ndarray float64):  
            - Current parameter set for optimisation.  
        - Pfix (numpy.ndarray float64):  
            - Same length as Pinitial. Shows if parametr will be optimised in algorithm. Fixed = 0, Free = 1.  
        - ub (numpy.ndarray float64):  
            - Upper bound for all parameters.  
        - lb (numpy.ndarray float64):  
            - Lower bound for all parameters.         - eval_step (float64):  
            - The evaluation step size for calculation of the gradient.  
        - costfn (function):  
            - Python or other callable function with appropriate wrapper for calculating single valued cost function output.  
        - current (float64):  
            - The current value of the cost function.  
        - args (Tuple):  
            - Collection of extra arguements used by the cost function.  

    Returns:
        
        - gradvec (numpy.ndarray float64):  
            - Gradient of cost function at each parameter. Length = len(Precon).
    '''
    from numpy import zeros_like
    
    gradvec = zeros_like(Precon)
    
    for i in range(0,len(Precon)):
        if Pfix[i] == 0:
            gradvec[i] = 0
        if Pfix[i] == 1:
            if Precon[i] > ub[i]-eval_step:
                Precon[i] -= eval_step
                new = costfn(Precon, Pmax, args)
                Precon[i] += eval_step
                gradvec[i] = - (new-current)/eval_step

            elif Precon[i] < lb[i]+eval_step:
                Precon[i] += eval_step
                new = costfn(Precon, Pmax, args)
                Precon[i] -= eval_step
                gradvec[i] = + (new-current)/eval_step
            elif lb[i] != 0 and 0 < Precon[i] < eval_step:
                Precon[i] = eval_step
                new_p = costfn(Precon, Pmax, args)
                Precon[i] = -eval_step
                new_n = costfn(Precon, Pmax, args)
                Precon[i] = 0
                gradvec[i] = (new_p-new_n)/(2*eval_step)
            elif lb[i] != 0 and -eval_step < Precon[i] < 0:
                Precon[i] = eval_step
                new_p = costfn(Precon, Pmax, args)
                Precon[i] = -eval_step
                new_n = costfn(Precon, Pmax, args)
                Precon[i] = 0
                gradvec[i] = (new_p-new_n)/(2*eval_step)
            else:
                Precon[i] += eval_step
                new_p = costfn(Precon, Pmax, args)
                Precon[i] -= 2*eval_step
                new_n = costfn(Precon, Pmax, args)
                Precon[i] += eval_step
                gradvec[i] = (new_p-new_n)/(2*eval_step)

    return gradvec

def step_scaling(gradvec, Pfix):
    
    ''' Normalise the gradient wrt to each parameter for use in backtracking algorithm.
    
    Parameters:
        
        - gradvec (numpy.ndarray float64):  
            - Gradient of cost function at each parameter. Length = len(Precon).  
        - Pfix (numpy.ndarray float64):  
            - Same length as Pinitial. Shows if parametr will be optimised in algorithm. Fixed = 0, Free = 1.  

    Returns:
        
        - scaled_step = (proportional_step * step_direction) ((numpy.ndarray float64)):  
            - Parameter step with correct direction for descent normalised to be maximum of abs(1).
    '''
    
    from numpy import clip, divide, zeros_like
    
    proportional_step = Pfix * clip((abs(gradvec) / max(abs(gradvec))), 0, 1)
    step_direction = - divide(gradvec, abs(gradvec), out=zeros_like(gradvec), where=abs(gradvec)!=0) 
    
    return proportional_step * step_direction

def backtracking_linesearch(Precon, Pmax, ls_step, scaled_step, ls_factor, ub, lb, costfn, current, args):
    
    '''Backtracking line search to find new best parameter set.
    
    Parameters:
        
        - Precon (numpy.ndarray float64):  
            - Current parameter set for optimisation.  
        - ls_step (float64):  
            - Initial step taken in each parameter, scaled using scaled_step.  
        - scaled_step (numpy.ndarray float64):  
            - Parameter step with correct direction for descent normalised to be maximum of abs(1).  
        - ls_factor (float64):  
            - Factor that reduces ls_step every iteration in the line search.  
        - ub (numpy.ndarray float64):  
            - Upper bounds for every parameter in Pinitial.  
        - lb (numpy.ndarray float64):  
            - Lower bounds for every parameter in Pinitial.  
        - costfn (function):  
            - Python or other callable function with appropriate wrapper for calculating single valued cost function output.  
        - current (float64):  
            - The current value of the cost function.  
        - args (Tuple):  
            - Collection of extra arguements used by the cost function.  
    
    Returns:
        
        - Pupdate (numpy.ndarray float64):  
            - New optimal parameter set.
    '''
    from numpy import isnan, heaviside
    from numpy import any as npany
    
    diff = 1
    
    while diff > 0 or isnan(diff):
        
        Pupdate = Precon + ls_step*scaled_step
    
        Pupdate[Pupdate>ub]=ub[Pupdate>ub]
        Pupdate[Pupdate<lb]=lb[Pupdate<lb]
    
        rescale_factors = []
        
        if npany((Pupdate*Precon) < 0):
            for i in range(0,len(Pupdate)):
                if heaviside(Pupdate[i] * Precon[i],1)==0 :
                    rescale_factors += [abs(Precon[i])/abs(Precon[i]-Pupdate[i])]
            Pupdate = Precon + min(rescale_factors)*ls_step*scaled_step
        else:
            rescale_factors = [0,0]
        
# =============================================================================
#         
#         while len(Pupdate[Pupdate > ub]) != 0 or len(Pupdate[Pupdate < lb]) != 0:
#             ls_step *= ls_factor
#             Pupdate = Precon + ls_step*scaled_step
# =============================================================================
# =============================================================================
#         if npany((Pupdate*Precon) < 0):
#             for i in range(0,len(Pupdate)):
#                 if heaviside(Pupdate[i] * Precon[i],1)==0 :
#                     Pupdate[i] = 0
# 
# =============================================================================
        updatecost = costfn(Pupdate, Pmax, args)
        diff = updatecost - current
        
        if diff <= 0:
            break
        ls_step *=  min(rescale_factors)
        ls_step *= ls_factor
    
    return Pupdate

def write_log(cost_file, Plist_file, Precon, current, updatecost, timestart):
    
    '''Writes the updated cost function and cost function changes to file, and tracks current parameter set.
    
    Parameters:
        
        - cost_file (.txt file):  
            - Text file tracking cost function values and time elapsed.  
        - Plist_file (.txt file):  
            - Test file tracking every iteration of parameter sets.  
        - Precon (numpy.ndarray float64):  
            - Current parameter set for optimisation.  
        - current (float64):  
            - The previous value of the cost function.  
        - updatecost (float64):  
            - The updated value of the cost function.  
        - timestart (float64):  
            - The starting time of the optimisation (absolute s).  
    
    Returns:
    
        - None
    '''
    
    from numpy import savetxt
    from time import perf_counter
    
    timenow = perf_counter()-timestart
    
    with open(cost_file,'a') as f:
        f.write("\n"+str(updatecost)+"\t"+str(timenow)+"\t"+str(current-updatecost))

    with open(Plist_file,'ab') as f:
        f.write(b'\n')
        savetxt(f, Precon, fmt='%1.5f', delimiter=',',newline=',')
    
    return

def open_log(fdir, Pinitial, Pinvmax):
    
    '''Creates log file instances and saves the list of true answers.
    
    Parameters:
        
        - fdir (string):  
            - Path to enclosing directory to save logging files.  
        - Pinitial (numpy.ndarray float64):   
            - Initial parameter set for optimisation.  
        - Pinvmax (numpy.ndarray float64):   
            - Maximum parameter values for optimisation parameter stack.  
    
    Returns:
        
        - cost_log (string):   
            - Path to file for cost function value logging.  
        - Plist_log (string):   
            - Path to file for parameter list logging.  
    '''
    
    from numpy import savetxt
        
    cost_log = '{}/costFn.txt'.format(fdir)
    Plist_log = '{}/Plist.txt'.format(fdir)
    Pinvmax_log = '{}/Pinvmax.txt'.format(fdir)
    
    with open(Plist_log,'wb') as f:
      savetxt(f, Pinitial, fmt='%1.5f', delimiter=',',newline=',')
    with open(Pinvmax_log,'wb') as f:
      savetxt(f, Pinvmax, fmt='%1.5f', delimiter=',',newline=',')
      
    return (cost_log, Plist_log)

def check_exitcond(cost_track, gtol, Pchange, Pecond, Pfix, updatecost, pastcost):
    
    '''Checks is the exit conditions for the algorithm have been meet at the end of each iteration.
    
    Parameters:
        
        - cost_track (numpy.ndarray float64):    
            - Array of the previous 4 iterations cost function values.  
        - gtol (int):    
            - Counter of the number of times the cost function has improved by below 1e-11.  
        - Pchange (numpy.ndarray float64):  
            - Change of each parameter at current update, in the normalised unitless picture.  
        - Pecond (numpy.ndarray float64):  
            - Lowest update parameter change acceptable for each variable. Lower than this for a certain number of parameters terminates algorithm.
        - updatecost (float64):    
            - The updated value of the cost function.  
        - pastcost (float64):    
            - The previous value of the cost function.  
        
    Returns:
        
        - exit_cond (None or string):    
            - Exit condition message or None type which keep optimisation running.
    '''
    from numpy import max as npmax
    from numpy import finfo
    cost_track[0:-1]=cost_track[1:]
    cost_track[-1]=updatecost
    
    if npmax(Pchange-Pecond) < 0:
        exit_cond = "Change below exit condition."
        return cost_track, gtol, exit_cond
    
    if (pastcost-updatecost)<1e-11:
        gtol += 1
    elif (pastcost-updatecost)>1e-11:
        gtol = 0
        
    exit_cond = None
    
    if (cost_track[0] == cost_track[1] == cost_track[2] == cost_track[3]):
        exit_cond = "No cost reduction for {} iterations".format(int(len(cost_track)))
        return cost_track, gtol, exit_cond
    
    if gtol == 15:
        exit_cond = "Reduction of cost function below threshold (1e-11). Exceeded {} times.".format(gtol)
        return cost_track, gtol, exit_cond
    
    if pastcost-updatecost < finfo(float).eps:
        exit_cond = 'Cost reduced below machine precision'
        return cost_track, gtol, exit_cond
    
    return cost_track, gtol, exit_cond

def Pstack(list_of_params):
    
    ''' Concatenate listed 1d arrays provided into order of appearance '''
    
    from numpy import append, empty
    
    P = empty((0))
    
    for entry in list_of_params:
        P = append(P,entry)
        
    return P
        
def extend(template, value):
    
    ''' Produce an array equal in length to template with all entries equal to value.  '''
    
    from numpy import ones, asarray
    template = asarray(template)
    return ones((template.size)) * value

def multires_setup_1d2c_flow(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args):
    
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_flow_1d2c
    from TKfunctions.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import linspace, mean,array
    from skimage.measure import block_reduce

    
    C_full, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic = args
    
    dx = array((12,6,3,2,1))
    Lx = sysdim[0]
    Nx = (Lx/dx).astype(int)
    
    meas_res[0] = dx[0]
    args[3] = meas_res
    
    C_meas = block_reduce(C_full, block_size = (int(Nx[-1]/Nx[0]),1), func = mean, cval = mean(C_full))
    args[0] = C_meas
    
    for i in range(0,len(dx)):
    
        Pfinal, cost_value, Pmax = simple_gradient_descent(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args)
               
        if i+1 != len(dx):
            output = param_extract_flow_1d2c(Pfinal, Nx[i])
            econd = param_extract_flow_1d2c(Pecond, Nx[i])
            max_vals = param_extract_flow_1d2c(Pmax, Nx[i])
            lbnds =  param_extract_flow_1d2c(lb, Nx[i])
            ubnds = param_extract_flow_1d2c(ub, Nx[i])
            fix = param_extract_flow_1d2c(Pfix, Nx[i])
            
            fa1, fv1, F, Da, Dv, va,va_frac,vv, v, Jpa,Jpv, Jna, Jnv = output
            fa1_econd, fv1_econd, F_econd, Da_econd, Dv_econd, va_econd,vaf_econd,vv_econd, v_econd, Jpa_econd,Jpv_econd, Jna_econd, Jnv_econd = econd
            fa1_max, fv1_max, F_max, Da_max, Dv_max, va_max,vaf_max,vv_max, v_max, Jpa_max,Jpv_max, Jna_max, Jnv_max= max_vals
            fa1_lb, fv1_lb, F_lb, Da_lb, Dv_lb, va_lb,vaf_lb,vv_lb, v_lb, Jpa_lb,Jpv_lb, Jna_lb, Jnv_lb = lbnds
            fa1_ub, fv1_ub, F_ub, Da_ub, Dv_ub, va_ub,vaf_ub,vv_ub, v_ub, Jpa_ub,Jpv_ub, Jna_ub, Jnv_ub = ubnds
            fa1_fix, fv1_fix, F_fix, Da_fix, Dv_fix, va_fix,vaf_fix,vv_fix, v_fix, Jpa_fix,Jpv_fix, Jna_fix, Jnv_fix = fix

            meas_res[0] = dx[i+1]
            x_old = linspace(0,Lx,Nx[i]+1)
            x_new = linspace(0,Lx,Nx[i+1]+1)
            xc_old = linspace(dx[i],Lx-dx[i],Nx[i])
            xc_new = linspace(dx[i+1],Lx-dx[i+1],Nx[i+1])
            
            F_new = interpolate(F, xc_old, xc_new)
            vaf_new = interpolate(va_frac, xc_old, xc_new)
            v_new = interpolate(v, xc_old, xc_new)
            Da_new = interpolate(Da, x_old, x_new)
            Dv_new = interpolate(Dv, x_old, x_new)
            
            F_econd = interpolate(F_econd, xc_old, xc_new)
            vaf_econd = interpolate(vaf_econd, xc_old, xc_new)
            v_econd = interpolate(v_econd, xc_old, xc_new)
            D_econd = interpolate(Da_econd, x_old, x_new)
            
            F_max = interpolate(F_max, xc_old, xc_new)
            vaf_max = interpolate(vaf_max, xc_old, xc_new)
            v_max = interpolate(v_max, xc_old, xc_new)
            D_max = interpolate(Da_max, x_old, x_new)
            
            F_lb = interpolate(F_lb, xc_old, xc_new)
            vaf_lb = interpolate(vaf_lb, xc_old, xc_new)
            v_lb = interpolate(v_lb, xc_old, xc_new)
            D_lb = interpolate(Da_lb, x_old, x_new)
            
            F_ub = interpolate(F_ub, xc_old, xc_new)
            vaf_ub = interpolate(vaf_ub, xc_old, xc_new)
            v_ub = interpolate(v_ub, xc_old, xc_new)
            D_ub = interpolate(Da_ub, x_old, x_new)
            
            F_fix = interpolate(F_fix, xc_old, xc_new)
            vaf_fix = interpolate(vaf_fix, xc_old, xc_new)
            v_fix = interpolate(v_fix, xc_old, xc_new)
            D_fix = interpolate(Da_fix, x_old, x_new)
            
            meas_res[0] = dx[i+1]
            args[3] = meas_res
    
            C_meas = block_reduce(C_full, block_size = (int(Nx[-1]/Nx[i+1]),1), func = mean, cval = mean(C_full))
            args[0] = C_meas
            
            Pinitial = Pstack( (fa1, fv1, F_new, Da_new, Dv_new, vaf_new, v_new, Jpa,Jpv, Jna, Jnv) )
            Pecond = Pstack( (fa1_econd, fv1_econd, F_econd, D_econd, D_econd, vaf_econd, v_econd, Jpa_econd,Jpv_econd, Jna_econd, Jnv_econd) )
            Pmax = Pstack( (fa1_max, fv1_max, F_max, D_max, D_max, vaf_max, v_max, Jpa_max,Jpv_max, Jna_max, Jnv_max) )
            lb = Pstack( (fa1_lb, fv1_lb, F_lb, D_lb, D_lb, vaf_lb, v_lb, Jpa_lb,Jpv_lb, Jna_lb, Jnv_lb) )
            ub = Pstack( (fa1_ub, fv1_ub, F_ub, D_ub, D_ub, vaf_ub, v_ub, Jpa_ub,Jpv_ub, Jna_ub, Jnv_ub) )
            Pfix = Pstack( (fa1_fix, fv1_fix, F_fix, D_fix, D_fix, vaf_fix, v_fix, Jpa_fix,Jpv_fix, Jna_fix, Jnv_fix) )
            
            print("Interpolated to next resolution")
        if i+1 == len(dx):
            break
    
    return Pfinal, cost_value, Pmax

def multires_setup_1d2c_vel(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, fdir, args):
    
    from TKfunctions.forward_models.onedim.fm_utilities import param_extract_vel_1d2c_msres as pextract
    from TKfunctions.forward_models.onedim.fm_utilities import interp_linear as interpolate
    from numpy import linspace, mean, append, savez_compressed, add, arange
    from skimage.measure import block_reduce
    from sigfig import round as sfround

    C_full, fitmod, fmod_setup, meas_res, sysdim, tsamp, comp, dtsim, pic,step, dx = args
    
    Lx = sysdim[0]
    Nx = (Lx/dx).astype(int)
    dx_sim = 0.01
    Nx_sim = int(Lx/dx_sim)
    meas_res[0] = dx[0]
    dy = meas_res[1]
    dz = meas_res[2]
    args[3] = meas_res
    
    Mass_fine = C_full*(dx_sim*dy*dz * 1e-3)
    Mass_meas = add.reduceat(Mass_fine, arange(0, len(Mass_fine[:,0]), int(dx[0]/dx_sim))) 
    C_meas = Mass_meas / (dx[0]*dy*dz*1e-3)
    args[0] = C_meas
    args[7] = dtsim[0]
    
    for i in range(0,len(dx)):
    
        Pfinal, cost_value, Pmax = simple_gradient_descent(Pinitial, Pmax, Pfix, Pecond, costfn, Niter, ub, lb, log, args)
        output = pextract(Pfinal, Nx[i])
        econd = pextract(Pecond, Nx[i])
        max_vals = pextract(Pmax, Nx[i])
        lbnds =  pextract(lb, Nx[i])
        ubnds = pextract(ub, Nx[i])
        fix = pextract(Pfix, Nx[i])
        
        ua, uv, Kva, Da, Dv, Ja, Jpa_frac, Jv, Jpv_frac = output
        ua_econd, uv_econd, Kva_econd, Da_econd, Dv_econd, Ja_econd, Jpa_f_econd, Jv_econd, Jpv_f_econd = econd
        ua_max, uv_max, Kva_max, Da_max, Dv_max, Ja_max, Jpa_f_max, Jv_max, Jpa_f_max = max_vals
        ua_lb, uv_lb, Kva_lb, Da_lb, Dv_lb, Ja_lb, Jpa_f_lb, Jv_lb, Jpv_f_lb = lbnds
        ua_ub, uv_ub, Kva_ub, Da_ub, Dv_ub, Ja_ub, Jpa_f_ub, Jv_ub, Jpv_f_ub = ubnds
        ua_fix, uv_fix, Kva_fix, Da_fix, Dv_fix, Ja_fix, Jpa_f_fix, Jv_fix, Jpv_f_fix = fix
        
        savez_compressed('{}/FinalOptimisationValues_dx_{}cm'.format(fdir,dx[i]), cost_value=cost_value,max_vals=max_vals, Pfinal=Pfinal,Pmax=Pmax, output=output, i=i)
        
        if i+1 != len(dx):

            meas_res[0] = dx[i+1]
            xc_old = linspace(dx[i],Lx-dx[i],Nx[i])
            xc_new = linspace(dx[i+1],Lx-dx[i+1],Nx[i+1])
            x_old = linspace(0,Lx,Nx[i]+1)
            x_new = linspace(0,Lx,Nx[i+1]+1)
            
            if Nx[i]==1:
                ua_new = append(ua,ua)
                uv_new = append(uv,uv)
                Da_new = append(Da,Da)
                Dv_new = append(Dv,Dv)
                Kva_new = append(Kva,Kva)
                
                ua_econd = append(ua,ua)
                uv_econd = append(uv,uv)
                Da_econd = append(Da,Da)
                Dv_econd = append(Dv,Dv)
                Kva_econd = append(Kva,Kva)
                
                u_max = append(ua_max,ua_max)
                D_max = append(Da_max,Da_max)
                Kva_max = append(Kva_max,Kva_max)
                
                u_lb = append(ua_lb,ua_lb)
                D_lb = append(Da_lb,Da_lb)
                Kva_lb = append(Kva_lb,Kva_lb)
                
                u_ub = append(ua_ub,ua_ub)
                D_ub =append(Da_ub,Da_ub)
                Kva_ub = append(Kva_ub,Kva_ub)
                
                u_fix = append(ua_fix,ua_fix)
                D_fix = append(Da_fix,Da_fix)
                Kva_fix = append(Kva_fix,Kva_fix)
            
            else:
                ua_new = interpolate(ua, x_old, x_new)
                uv_new = interpolate(uv, x_old, x_new)
                Da_new = interpolate(Da, x_old, x_new)
                Dv_new = interpolate(Dv, x_old, x_new)
                Kva_new = interpolate(Kva, xc_old, xc_new)
                
                ua_econd = interpolate(ua_econd, x_old, x_new)
                uv_econd = interpolate(uv_econd, x_old, x_new)
                Da_econd = interpolate(Da_econd, x_old, x_new)
                Dv_econd = interpolate(Dv_econd, x_old, x_new)
                Kva_econd = interpolate(Kva_econd, xc_old, xc_new)
                
                u_max = interpolate(ua_max, x_old, x_new)
                D_max = interpolate(Da_max, x_old, x_new)
                Kva_max = interpolate(Kva_max, xc_old, xc_new)
                
                u_lb = interpolate(ua_lb, x_old, x_new)
                D_lb = interpolate(Da_lb, x_old, x_new)
                Kva_lb = interpolate(Kva_lb, xc_old, xc_new)
                
                u_ub = interpolate(ua_ub, x_old, x_new)
                D_ub = interpolate(Da_ub, x_old, x_new)
                Kva_ub = interpolate(Kva_ub, xc_old, xc_new)
                
                u_fix = interpolate(ua_fix, x_old, x_new)
                D_fix = interpolate(Da_fix, x_old, x_new)
                Kva_fix = interpolate(Kva_fix, xc_old, xc_new)
            meas_res[0] = dx[i+1]
            args[3] = meas_res
            args[7] = dtsim[i+1]
    
            Mass_meas = add.reduceat(Mass_fine, arange(0, len(Mass_fine[:,0]), int(dx[i+1]/dx_sim))) 
            C_meas = Mass_meas / (dx[i+1]*dy*dz*1e-3)
            args[0] = C_meas
            
            Pinitial = Pstack( (ua_new, uv_new, Kva_new, Da_new, Dv_new, Ja, Jpa_frac, Jv, Jpv_frac) )
            Pecond = Pstack( (ua_econd, uv_econd, Kva_econd, Da_econd, Dv_econd, Ja_econd, Jpa_f_econd, Jv_econd, Jpv_f_econd) )
            Pmax = Pstack( (u_max, u_max, Kva_max, D_max, D_max, Ja_max, Jpa_f_max, Jv_max, Jpa_f_max) )
            lb = Pstack( (u_lb, u_lb, Kva_lb, D_lb, D_lb, Ja_lb, Jpa_f_lb, Jv_lb, Jpv_f_lb) )
            ub = Pstack( (u_ub, u_ub, Kva_ub, D_ub, D_ub, Ja_ub, Jpa_f_ub, Jv_ub, Jpv_f_ub) )
            Pfix = Pstack( (u_fix, u_fix, Kva_fix, D_fix, D_fix, Ja_fix, Jpa_f_fix, Jv_fix, Jpv_f_fix) )
            
            dt_current = dtsim[i+1]
            args[7] = dt_current
            print("Interpolated to next resolution")
        else:
            break
    
    return Pfinal, cost_value, Pmax
