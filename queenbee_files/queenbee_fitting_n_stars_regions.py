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
import Fitting_and_pdfs as fap

"""
1. Defining Input Parameters
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

# number to which the dimensions of the original image is reduced to cal chi-squared
num_splits = 5

# different populations of hot stars considered 
hot_stars_arr = np.arange(0, 11)/10

# would you like to discard foreground stars completely or do something more complicated
discard_forground_stars = False
foreground_cutoff = 1.1

# discards background stars from the effective FoV
if discard_forground_stars:
    x_pos, y_pos, mag_H, mag_Ks = ssfm.discardForegroundStars(x_pos, y_pos, mag_H, mag_Ks, foreground_cutoff)
    num_stars = len(mag_Ks)

# start pixels of a small region of the sky
x_start, y_start = 0, 0

# start pixels in the original file
x_start_og, y_start_og = 1000, 2500


"""
2. Fitting regions with n-stars
"""
# model all regions with x neighbouring stars
n_stars = 11

resulting_params_all = np.zeros((0, len(hot_stars_arr), 2))

# define the limits of the region to be cropped
limits = np.load(save_data_crop_dir + '%d_stars/limits.npy'%(n_stars))

# associate spectra to those stars
stars_with_n_neighbours = np.where(total_neighbours == n_stars)


for i, stars_idx in enumerate(stars_with_n_neighbours[0]):
    # arr to save the chi-sq val for a given hot star distribution, and the best template no. for this chi-sq
    resulting_params = np.zeros((0, 2))
    print('\nRegion %d of %d:'%(i, len(stars_with_n_neighbours[0])))
    
    # get positions and magnitudes of the neighbouring stars
    x, y, mKs = star_neighbours[1][stars_idx][0], star_neighbours[1][stars_idx][1], star_neighbours[1][stars_idx][3]
    
    # load PSF matrix and data matrix for the region for the given number of neighbours
    data_mat = np.load(save_data_crop_dir + '%d_stars/data_mat_%d.npy'%(n_stars, i))
    data_PSF_mat = np.load(save_data_psf_crop_dir + '%d_stars/data_mat_PSF_%d.npy'%(n_stars, i))
        
    # reduce the dimensions of the matrix to cal. the minimizing statistic
    data_mat_grid, x_pixels = fap.reduceMatrixResolutions(data_mat, limits[i], num_splits)
    data_PSF_mat_grid = fap.reduceMatrixResolutions(data_PSF_mat, limits[i], num_splits)
    
    # iterating over every possible hot star population
    for j, hot_stars in enumerate(hot_stars_arr): 
        print('\nFitting for %d percent hot stars'%(hot_stars*100))
        
        # generate 2D matricies from the template library by associating x, y and mKs to spectra
        perms = np.load(template_dir+'%d_stars/%d_hot_stars/perm_arr.npy'%(n_stars, hot_stars*n_stars)) 
        
        # iterating over all the permutations for the given hot star population
        chi_sq_val, best_fit_perm_no = fap.generateTemplatesCalChiSquare(x, y, mKs, template_dir, save_data_psf_crop_dir, n_stars,\
                                                      hot_stars, perms, disperse_range, dispersion_angle,\
                                                      u_pix_arr, limits[i], data_mat, data_PSF_mat, \
                                                       data_mat_grid, i, x_FOV[stars_idx], num_splits=5)
        
        resulting_params = np.append(resulting_params, [[chi_sq_val, best_fit_perm_no]], axis=0)
        
    resulting_params_all = np.append(resulting_params_all, [resulting_params], axis=0)
    
# save the best chi-square values for the given region
np.save('Data/Chi_sq_vals/%d_stars_%d_regions.npy'%(n_stars, len(stars_with_n_neighbours[0])), resulting_params_all)