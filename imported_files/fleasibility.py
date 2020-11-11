# -*- coding: utf-8 -*-
"""fleasibility.py

##  Functions for studying the fleasibility of doing slitless spectroscopy in the Galactic Centre region

This py file builds on a lot of the function written in Slitless_spec_forward_modelling.py

**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 3rd November 2020


## 1. Imports
"""

import astropy.units as u
import astropy.io.fits as fits

from scipy.sparse import csr_matrix
import scipy.stats as stats
from scipy.interpolate import interp1d
import numpy as np
import os
import importlib
import math  

# generate random integer values
from random import seed
from random import randint

# plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# for manupilating spectra
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
from specutils import Spectrum1D

# to show progress during computations
import time

# for doing combinatorics with spectra
from itertools import permutations

# personal imports
import Simulating_Spectra as ss
import Slitless_spec_forward_modelling as ssfm
import plotting as pt

# for generating permutations of star configurations
from sympy.utilities.iterables import multiset_permutations
from sympy import factorial
"""

### 1. Functions for reading in the catalog

"""

def selectionCuts(x_pos, y_pos, mag_H, mag_Ks, cut_off_ll, pixel_factor):
    """
    Function to select valid stars for the entire central file
    """
    # selects real stars in the FOV
    x_pos, y_pos, mag_H, mag_Ks = ssfm.selectRealStars(x_pos, y_pos, mag_H, mag_Ks)
    
    # cuts off stars below a certain flux from the bottom
    x_pos, y_pos, mag_H, mag_Ks = ssfm.cutOffFlux(cut_off_ll, x_pos, y_pos, mag_H, mag_Ks)
    
    # discards stars within the NSC
    x_pos, y_pos, mag_H, mag_Ks  = ssfm.discardNSC(x_pos, y_pos, mag_H, mag_Ks, pixel_factor)
    return x_pos, y_pos, mag_H, mag_Ks

def printDivisors(n):
    """
    Function to calculate the divisors of integer n
    @Returns ::
    @divisors :: the divisors that are closed to the square root
    """  
    divisor, divisors_partner = [], []
    i = 1
    while i <= math.sqrt(n):          
        if (n % i == 0) :              
            # If divisors are equal, print only one 
            if (n / i == i) : 
                print(i) 
            else : 
                # Otherwise print both
                divisor.append(i)
                divisors_partner.append(n/i)                
        i = i + 1    
    return np.max(divisor), int(np.min(divisors_partner))

def createGrid(x_all, y_all, mag_H_all, mag_Ks_all, u_pix):
    """
    Function to calculate the forground population in every ~1.2'squared of the central file
    """
    # arr to save forground and background info
    foreground_stars, gc_stars = [], []
    
    # arrays that set the sizes of the grids
    steps_x = np.arange(np.min(x_all), np.max(x_all), u_pix)
    steps_y = np.arange(np.min(y_all), np.max(y_all), u_pix)
    
    # looping over every grid of size u_pix
    for i, x in enumerate(steps_x):
        for j, y in enumerate(steps_y):
            limits = ssfm.generateLimits(x, y, u_pix)
            _, _, mag_H, mag_Ks = ssfm.selectFOV(limits, x_all, y_all, mag_H_all, mag_Ks_all, print_msg=False)
            
            # save the info
            foreground_stars.append(len(np.where( (mag_H - mag_Ks)<1.1 )[0]))
            gc_stars.append(len(mag_Ks))
            
    return foreground_stars, gc_stars, [steps_x, steps_y]

def calPermutations(hot_stars, num_stars, type_id_arr, num_perms):
    """
    Function to calculate the number of permutations i.e. templates required for a given hot_star population
    @hot_stars :: percent of hot stars in the FOV
    @num_stars :: total number of stars in the field of view
    """
    # Load the 10 random spectra with added LSF, which can be associated with the stars in the FOV.
    flux_LSF2D, params = np.load('Data/flux_K2D_LSF.npy'), np.load('Data/params.npy')
    waves_k = np.load('Data/waves_k.npy')
    

    # division of stars chosen in FOV, considering 10 temperatures with 10 log g's
    stars_divide = ssfm.decideNumHotStars(hot_stars=hot_stars)
    
    # type_id is the required array to compute all possible required permutations
    _, type_id = ssfm.associateSpectraToStars(waves_k, stars_divide, num_stars, \
                                      flux_LSF2D, params)    
    type_id_arr.append(type_id)
    
    m = hot_stars*num_stars
    n = num_stars
    num_perms.append(factorial(n)/(factorial(m)* factorial(n-m) ) )
    return type_id_arr, num_perms