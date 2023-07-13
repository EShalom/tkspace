#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:15:14 2021

@author: pyess
"""

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
    return

def tidy(x, n):
    """Return 'x' rounded to 'n' significant digits."""
    import sys,math
    y=abs(x)
    if y <= sys.float_info.min: return 0.0
    return round( x, n-math.ceil(math.log10(y)) )

