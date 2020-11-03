# -*- coding: utf-8 -*-
"""fitting_and_pdfs.py

##  Functions for fitting the templates to the data, generating data, and plotting the chi-squared distribution for different configurations of stellar population


**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 2nd November 2020

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

# generate random integer values
from random import seed
from random import randint

# plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

import matplotlib

# for manupilating spectra
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
from specutils import Spectrum1D


# for doing combinatorics with spectra
from itertools import permutations

# personal imports
import Simulating_Spectra as ss
import Slitless_spec_forward_modelling as ssfm
import plotting as pt
"""

### 1. Functions for fitting the templates from the generated library

"""

def dataMatrix(template_dir, hot_stars, num_stars, u_pix, l_pix, x_pos, y_pos, disperse_range, dispersion_angle, templates_exist):
    """
    Function to build a data matrix from the template library for a given distribution of stars
    @template_dir :: directory where the templates are stored
    @hot_stars :: percent of hot stars in the FOV
    @num_stars :: number of stars in the FOV
    @l_pix, u_pix :: lower and upper limit in pixels defining the FOV
    @x_pos, y_pos :: the x and y position of the star  
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    @dispersion_angle :: variable sets the orientation of the dispersion
    @templates_exist :: boolean to check if the templates exist for the given population
    
    Returns ::
    @data_LSF_2Dmat :: 2darray containing the so-called data
    """
    # Load the 10 random spectra with added LSF, which can be associated with the stars in the FOV.
    flux_LSF2D, params = np.load('Data/flux_K2D_LSF.npy'), np.load('Data/params.npy')
    waves_k = np.load('Data/waves_k.npy')
    
    # division of stars chosen in FOV, considering 10 temperatures with 10 log g's
    stars_divide = ssfm.decideNumHotStars(hot_stars=hot_stars)
    x_disperse, y_disperse = pt.plotDispersedStars(x_pos, y_pos, l_pix, u_pix, \
                                               disperse_range, waves_k, \
                                               dispersion_angle)
    
    flux_k2D, type_id = ssfm.associateSpectraToStars(waves_k, stars_divide, num_stars, \
                                      flux_LSF2D, params)

    # generate all possible permutations that exist for the given distribution of stars
    if templates_exist:
        perms_arr = ssfm.constructSpectralTemplates(u_pix, y_disperse, x_disperse, \
                                                    type_id, flux_k2D)

    # building a data image that considers all possible (distinguishable) permuations
    perms_arr = np.load(template_dir+'hot_stars%d/perm_arr.npy'%(hot_stars*num_stars))
    data_LSF_2Dmat, perm_no = ssfm.constructDataImage(perms_arr, u_pix, flux_k2D, \
                                            y_disperse, x_disperse)
    return data_LSF_2Dmat, perm_no

def fitTemplateLibrary(template_dir, hot_stars_arr, num_stars, data_LSF_2Dmat, num_splits, u_pix):
    """
    Function to test all possible permutations for all configurations (stellar populations) of hot stars to the data image
    @template_dir :: directory where the templates are stored
    @hot_stars_arr :: arr holding info about the different configurations of hot stars considered
    @num_stars :: number of stars in the FOV
    @data_LSF_2Dmat :: 2darray containing the so-called data
    @num_splits :: reduced dimension of the flux matrix
    @u_pix :: upper limit in pixels defining the FOV
    
    """
    # store the chi-square value for every template
    chi_square3D = []
    # reduce the dimensions of the matrix to cal. the minimizing statistic 
    data_vals_2Darr = ssfm.splitFOVtoGrid(data_LSF_2Dmat, num_splits)
    
    for idx, stars in enumerate(hot_stars_arr):
        print('\n\nFor %d hot stars:\n'%(stars*num_stars))
        perms_arr = np.load(template_dir+'hot_stars%d/perm_arr.npy'%(stars*num_stars))
    
        # cal the chi-squared and the minimum chi-squared configuration
        chi_squared_arr = ssfm.determineBestFit(u_pix, num_stars, perms_arr, \
                                                data_vals_2Darr, num_splits)
        chi_square3D.append(chi_squared_arr)            
    return chi_square3D

