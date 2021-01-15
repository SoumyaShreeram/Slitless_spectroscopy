# -*- coding: utf-8 -*-
"""Slitless_spec_forward_modelling.py

##  Functions for selecting stars and forward modeling the Galactic centre region in Slitless-Spectroscopy mode. The file is categorized in the following way:

1. Functions for reading in the catalog
2. Functions for selection of stars

**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 9th October 2020

## 1. Imports
"""
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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

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
import Fitting_and_pdfs as fap

"""
01. Defining Input parameters
"""
# FOV in pixels
l_pix, u_pix = 0, 1128

# loads the effective FoV
x_pos, y_pos, mag_H, mag_Ks = np.load('Data/selected_FOV.npy')
selected_c_pxls = np.load('Data/selected_central_pixels.npy')

# directory paths
template_dir = 'Data/Template_library/'
save_data_crop_dir = 'Data/Cropped_Data_Images/'
save_data_psf_crop_dir = 'Data/Cropped_Data_PSF_Images/'
        
# set the dispersion range of the spectra
disperse_range = ss.defineDispersionRange(r=4000, lambda0=2.2, band_width=0.4)
print('The size of dispersion is set to be %d pixels'%disperse_range)

# effective FoV dimensions
u_pix_arr = [int(u_pix+disperse_range), u_pix]

dispersion_angle = 0 # degrees

# width of influence of the PSF on neighbouring stars (pixels)
width=3

# different populations of hot stars considered 
hot_stars_arr = np.arange(0, 11)/10

# would you like to discard foreground stars completely or do something more complicated
discard_forground_stars = False

# discards background stars from the effective FoV
if discard_forground_stars:
    foreground_cutoff = 1.1
    x_pos, y_pos, mag_H, mag_Ks = ssfm.discardForegroundStars(x_pos, y_pos, mag_H, mag_Ks, foreground_cutoff)
    num_stars = len(mag_Ks)

# start pixels of a small region of the sky
x_start, y_start = 0, 0

# start pixels in the original file
x_start_og, y_start_og = 1000, 2500

"""
02. Extracting information on the neighbouring population of stars for the concerned FOV

"""
# these arrays that hold information about the location and number of neighbours
star_neighbours, num_neighbours, x_FOV, y_FOV = stl.findStarAndNeighbours(x_pos, y_pos, mag_H,\
                                                            mag_Ks, disperse_range,\
                                                            width, selected_c_pxls,\
                                                           [x_start, y_start], u_pix)

# arrays extract information about the no. of neighbours in total & outside FOV
neighbours_outside_FOV = np.array(num_neighbours[1])-np.array(num_neighbours[0])
total_neighbours = np.array(num_neighbours[1])

"""
03. Gather the best fitting templates
"""
# gather the info from the 'data' image
num_splits = 5
for n_stars in [11, 12, 13, 14, 15, 16]:	
	stars_with_n_neighbours = np.where(total_neighbours == n_stars)

	stellar_types_data_arr, target_star_type, target_star_idx = stl.findStellarTypesDataStars(n_stars, hot_stars_arr,\
	                                                                                          stars_with_n_neighbours,\
	                                                                                          star_neighbours, x_pos,\
	                                                                                          y_pos, x_FOV, y_FOV)    

	# Loading the chi-squares for all regions and all templates with n_stars
	resulting_params_all = np.load('Data/Chi_sq_vals/%d_stars_%d_regions.npy'%(n_stars, len(stars_with_n_neighbours[0])), allow_pickle=True)                                                                  

	accuracy_vals = []

	num_temps_arr = [12, 25, 50, 75, 100, 125, 150, 175]

	for j, param in enumerate(num_temps_arr):
		method = ['find_N_smallest_vals', param]
	    
	    hot_star_counts, target_star_prediction_all = stl.getTargetStarPrediction(hot_stars_arr, resulting_params_all, stars_with_n_neighbours, \
		template_dir, n_stars, target_star_idx, print_msg=False, method=method)

		hot_probability, cold_probability = pt.plotTargetStarPrediction('ax[j+1]', hot_star_counts, target_star_prediction_all, 'hatch[j]', method[1], n_stars, plot_fig=False)

	    #pt.plotChiSqPredictions(ax[0], cold_probability, hot_probability, linestyle[j], chi_sq_cut, n_stars)

		accuracy = fap.evaluateAccuraryNstars(hot_probability, cold_probability, target_star_type)
		accuracy_vals.append(accuracy[0])
		
		print('The accuracy for %d star regions (%d lowest templates) is %.2f %s'%(n_stars, param, accuracy[0]*100, '%'))

	np.save('Data/Chi_sq_vals/%d_stars_accuracy_vals_%d_pixel_scale.npy'%(n_stars, num_splits), accuracy_vals)