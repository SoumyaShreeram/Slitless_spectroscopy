import astropy.units as u
import astropy.io.fits as fits

import math
from scipy.sparse import csr_matrix
import scipy.stats as stats
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

# to show progress during computations
from time import sleep
import sys

import matplotlib

# for doing combinatorics with spectra
from itertools import permutations
from more_itertools import distinct_permutations

sys.path.append('imported_files/')
import Simulating_Spectra as ss
import plotting as pt
import Slitless_spec_forward_modelling as ssfm
import star_by_star_template_library as stl

"""
1. Defining Input Parameters
"""
# FOV in pixels
l_pix, u_pix = 0, 1128

x_pos, y_pos, mag_H, mag_Ks = np.load('Data/selected_FOV.npy')
selected_c_pxls = np.load('Data/selected_central_pixels.npy')
num_stars = len(mag_Ks)

template_dir = 'Data/Template_library/'
hot_stars = 0.1

# set the dispersion range of the spectra
disperse_range = ss.defineDispersionRange(r=4000, lambda0=2.2, band_width=0.4)
print('The size of dispersion is set to be %d pixels'%disperse_range)

dispersion_angle = 0 # degrees

# width of influence of the PSF on neighbouring stars (pixels)
width=3

# number to which the dimensions of the original image is reduced to cal chi-squared
num_splits = int(u_pix/5)

# different populations of hot stars considered 
hot_stars_arr = np.arange(0, 11)/10

# Load the 10 random spectra with added LSF, which can be associated with the stars in the FOV.
flux_LSF2D, params = np.load('Data/flux_K2D_LSF_norm.npy'), np.load('Data/params.npy')
waves_k = np.load('Data/waves_k.npy')

# would you like to discard foreground stars completely or do something more complicated
discard_forground_stars = True
foreground_cutoff = 1.1

# start pixels of a small region of the sky
x_start, y_start = 1000, 2500

if discard_forground_stars:
    x_pos, y_pos, mag_H, mag_Ks = ssfm.discardForegroundStars(x_pos, y_pos, mag_H, mag_Ks, foreground_cutoff)
    print('Now for the whole central catalog: ')
    selected_c_pxls = ssfm.discardForegroundStars(selected_c_pxls[0], selected_c_pxls[1], selected_c_pxls[2], selected_c_pxls[3], foreground_cutoff)
    
"""
2. Understanding the neighbouring population of stars for the concerned FOV
"""
# these arrays that hold information about the location and number of neighbours
star_neighbours, num_neighbours = stl.findStarAndNeighbours(x_pos, y_pos, mag_H,\
                                                            mag_Ks, disperse_range,\
                                                            width, selected_c_pxls,\
                                                           [x_start, y_start], u_pix)

# arrays extract information about the no. of neighbours in total & outside FOV
neighbours_outside_FOV = np.array(num_neighbours[1])-np.array(num_neighbours[0])
total_neighbours = np.array(num_neighbours[1])
    
num_stars_arr = np.sort(np.unique(total_neighbours))

for num_stars in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
    for hot_stars in hot_stars_arr:
        print('\nGenerating templates for %d stars of which there exist %d hot stars...'%(num_stars, num_stars*hot_stars))
    
        # define the division of stars in the chosen FOV (hot or cold)
        stars_divide = ssfm.decideNumHotStars(hot_stars = hot_stars)

        flux_k2D, type_id = ssfm.associateSpectraToStars(waves_k, stars_divide, \
                                                         num_stars, flux_LSF2D, \
                                                         params, print_msg=False)
        # generate and save the templates
        stl.generateSaveTemplates(type_id, flux_LSF2D, flux_k2D, template_dir, num_stars, hot_stars)         