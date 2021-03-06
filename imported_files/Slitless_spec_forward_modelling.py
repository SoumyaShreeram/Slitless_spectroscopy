# -*- coding: utf-8 -*-
"""Slitless_spec_forward_modelling.py

##  Functions for selecting stars and forward modeling the Galactic centre region in Slitless-Spectroscopy mode. The file is categorized in the following way:

1. Functions for reading in the catalog
2. Functions for selection of stars

Note that the construction of data and template matricies are valid only for the 10 star model. For the generalized case refer to 'star_by_star_template_library.py'

**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 9th October 2020

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


# for doing combinatorics with spectra
from itertools import permutations

# personal imports
import Simulating_Spectra as ss
import plotting as pt
"""

### 1. Functions for reading in the catalog

"""

def readCatalogFile(filename, pixel_factor):
    """
    Function to read the magnitude and positions of the stars from the catalog file
    """
    central_pixels = np.loadtxt(filename)
    
    x_pos, y_pos = central_pixels[:, 0], central_pixels[:, 1] 
    mag_H, mag_Ks = central_pixels[:, 4], central_pixels[:, 6]
    return x_pos/pixel_factor, y_pos/pixel_factor, mag_H, mag_Ks

def decideNumHotStars(hot_stars):
    """
    Function to produce an array holding information about the stellar distribution
    @hot_stars :: number between 0.1-0.9 to define the percent of hot stars in the population
    """
    #TODO: try, catch to correct numbers not between 0.1 and 0.9
    # t_eff_arr = [12000, 11800, 10000, 8400, 7600, 6900, 5900, 5100, 4700, 3900]    
    stars_divide = [hot_stars, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1-hot_stars, 0.0, 0.0]
    return stars_divide

def generateHotStarArr(num_stars):
    hot_stars_arr = []
    for i in range(num_stars+1):
        hot_stars_arr.append(i/num_stars)
    return hot_stars_arr


def generateLimits(x_start, y_start, u_pix):
    limits = []
    if isinstance(u_pix, (int, float)):
        limits =  [x_start, x_start+u_pix, y_start, y_start+u_pix]
    # if FOV is a rectangle
    if isinstance(u_pix, (list, tuple, np.ndarray)):
        limits = [x_start, x_start+u_pix[0], y_start, y_start+u_pix[1]]
    return limits
     
"""

### 2. Functions for selection of stars

"""

def selectFOV(limits, x_pos, y_pos, mag_H, mag_Ks, print_msg=True):
    """
    Function selects stars in the given region of the sky
    @x_pos, y_pos, mag :: x_pos, y_pos, and magnitude of stars in the H and Ks band
    """   
    # select desired coordinates
    select_idx = np.where((x_pos>limits[0]) & (x_pos<limits[1]) & (y_pos>limits[2]) & (y_pos < limits[3]))
    
    if print_msg:
        print('Choosing %d stars from %d total stars.'%(len(select_idx[0]), len(mag_Ks)))
    return x_pos[select_idx], y_pos[select_idx], mag_H[select_idx], mag_Ks[select_idx]

def selectRealStars(x_pos, y_pos, mag_H, mag_Ks):
    """
    Function selects real stars in the FOV
    @x_pos, y_pos, mag :: x_pos, y_pos, and magnitude of stars in the H and Ks band
    """    
    select_idx = np.where((mag_H!=99) & (mag_Ks!=99))
    print("Selecting real stars...")
    return x_pos[select_idx], y_pos[select_idx], mag_H[select_idx], mag_Ks[select_idx]

def cutOffFlux(cut_off_ll, x_pos, y_pos, mag_H, mag_Ks):
    """
    Function cuts off stars below a certain flux from the bottom
    @x_pos, y_pos, mag :: x_pos, y_pos, and magnitude of stars in the H and Ks band
    @cut_off_ll :: cut-off limit on the magnitude (note: mag scale is -ve)
    """    
    select_idx = np.where(mag_Ks<cut_off_ll)
    print("Discarding stars with magnitude > %d."%cut_off_ll)
    return x_pos[select_idx], y_pos[select_idx], mag_H[select_idx], mag_Ks[select_idx]

def discardNSC(x_pos, y_pos, mag_H, mag_Ks, pixel_factor):
    """
    Function cuts off stars belonging to the NSC
    """
    # radius of the NSC (pixels)
    radius = 2414/pixel_factor
    
    # converting x-y coordinates to pixels
    d = ((x_pos - 20674/pixel_factor)**2+(y_pos - 8194/pixel_factor)**2)**0.5
    
    # discarding all stars outside the defined radius
    count = np.where(d >= radius)
    
    print('Discarding all stars withing the NSC...')
    return x_pos[count], y_pos[count], mag_H[count], mag_Ks[count]

def discardForegroundStars(x_pos, y_pos, mag_H, mag_Ks, foreground_cutoff):
    """
    Function to discard the foreground stars that pollute the Galactic center stars
    """
    count = np.where( (mag_H - mag_Ks)>1.1 )
    print('Discarding all the forground stars...')
    return x_pos[count], y_pos[count], mag_H[count], mag_Ks[count]

def mapToFOVinPixels(x_pos, y_pos, u_pix):
    """
    Function to map the r.a. and declination positions into pixels 
    @x_pos, y_pos, mag :: x_pos, y_pos, and magnitude of stars in the H and Ks band
    """  
    if isinstance(u_pix, (int, float)):
        funcX = interp1d([np.min(np.abs(x_pos)), np.max(np.abs(x_pos))], [0, u_pix-1])
        funcY = interp1d([np.min(y_pos), np.max(y_pos)],[0, u_pix-1]) 
        
    # if FOV is a rectangle
    if isinstance(u_pix, (list, tuple, np.ndarray)):
        funcX = interp1d([np.min(np.abs(x_pos)), np.max(np.abs(x_pos))], [0, u_pix[0]-1])
        funcY = interp1d([np.min(y_pos), np.max(y_pos)],[0, u_pix[1]-1])        
    return funcX(np.abs(x_pos)), funcY(y_pos)

def calculatedTotalStars(num_hot_st, num_cold_st, tot_stars):
    if int(num_hot_st)+int(num_cold_st) != tot_stars:
        diff1 = int(num_hot_st)-num_hot_st
        diff2 = int(num_cold_st)-num_cold_st
        
        if np.abs(diff1) > np.abs(diff2):
            num_hot_st = int(num_hot_st)+1
        else:
            num_cold_st = int(num_cold_st)+1
    return int(num_hot_st), int(num_cold_st)

def checkTotalStars(idx, star_percent, tot_stars):
    # for all the hot stars
    if idx < 5: 
        num_hot_st = star_percent*tot_stars
        num_cold_st = (1-star_percent)*tot_stars
        out_stars, _ = calculatedTotalStars(num_hot_st, num_cold_st, tot_stars)
    # reverse rols
    else:
        num_cold_st = star_percent*tot_stars
        num_hot_st = (1-star_percent)*tot_stars
        _, out_stars = calculatedTotalStars(num_hot_st, num_cold_st, tot_stars)
    return out_stars

def printHotColdStars(idx, num_stars):
    if idx < 5:
        print('Number of hot stars: %d'%num_stars)
    else:
        print('Number of cold stars: %d'%num_stars)
    return

def associateSpectraToStars(waves_k, stars_divide, tot_stars, flux_LSF2D, params, print_msg):
    """
    Function to associate the spectra of a certain type to the star
    @waves_k :: wavelength arr in the k-band
    @stars_divide :: arr that categorieses the #stars
    @tot_stars :: total number of stars in the FOV
    @flux_LSD2D :: flux array containins spectra for different spectral_types
    @params :: arr containing [t_eff, log_g, Fe_H, alpha, spectral_types]
    
    Returns 
    @flux_k2D :: 
    @type_id :: arr keeps track of the different types of stars with labels running from 0 to len(star_divide)
    """
    label_arr = np.arange(len(stars_divide))
    flux_k2D = np.zeros((0, len(waves_k)))
    type_id = []
    
    # to get the right result for the iteration
    #tot_stars = tot_stars+1
    
    for idx, star_percent in enumerate(stars_divide):        
        if star_percent*tot_stars != 0:
            num_hot_cold_stars = checkTotalStars(idx, star_percent, tot_stars)
            printHotColdStars(idx, num_hot_cold_stars)
            
            for sp in range(num_hot_cold_stars):
                    # keep track of the type of the star
                    type_id.append(label_arr[idx])
                    
                    # add the spectra for the star into the array
                    flux_k2D = np.append(flux_k2D, [flux_LSF2D[idx]], axis=0)
                    if print_msg:
                        print('%d stars at Teff = %s K, log g = %s'%(star_percent*tot_stars, params[idx][0], params[idx][1]))
            
        # if there are no stars with the chosen spectrum
        else:            
            if print_msg:
                print('%d stars at Teff = %s K, log g = %s'%(star_percent*tot_stars, params[idx][0], params[idx][1]))
    if print_msg:
                print('---------------------------------\n')
    return flux_k2D, type_id

def shuffleAlongAxis(arr_a2D, arr_b1D):
    """
    Function to shuffle matrix along a particular axis
    @arr :: preferable 2D arr
    @axis :: axis along which one would like to shuffle the arr (0 for rows, 1 for cols)
    """
    assert len(arr_a2D) == len(arr_b1D)
    p = np.random.permutation(len(arr_a2D))
    return arr_a2D[p], arr_b1D[p]

"""

### 3. Functions for generating a template library and choosing the best template

"""

def generateAllCombinations(type_id):
    "Function generates all possible distinguishable combinations of input array"
    perms = set(permutations(type_id))
    return list(perms)

def constructDataImage10Stars(perms, u_pix, flux_k2D, y_disperse, x_disperse):
    """
    Function to construct an arbitary 'data image' that considers a possible spectral combinations of stars for the 10 star model
    """
    flux_matrix2D = np.zeros((u_pix, u_pix))
    flux_PSF_2Dmatrix = np.load('Data/10_star_model/flux_PSF_2Dmatrix.npy')
    
    # choosing a random permutation number
    idx = ss.generateRandInt(0, len(perms)-1, 1)
    
    # reorder the flux arr for the given permutation
    flux_k2D_temp = flux_k2D[np.array(perms[idx[0]])]

    # construct the 2D matrix
    flux_matrix2D = ss.construct2DFluxMatrix(flux_matrix2D, y_disperse, x_disperse, \
                                     flux_k2D_temp, u_pix) 

    # add psf and noise
    flux_matrix2D = ss.addNoise(flux_matrix2D + flux_PSF_2Dmatrix, u_pix)  
    print('\nChoosing permutation no. %d out of total %d permutations'%(idx[0], len(perms)))
    return flux_matrix2D, idx[0]


def countHotStars(perms_arr):
    """
    Function to count the number of hot stars in the given template configuration
    @perms_arr :: 2D arr holding information about the possible permutations for a given configuration
    Returns ::
    @count_stars :: counts the number of hot stars (limit of this defintion can be changed) in the FOV
    """
    count_stars = 0
    for perm in perms_arr[0]:
        if perm == 0:
            count_stars += 1
    return count_stars

def constructSpectralTemplates10Stars(u_pix, y_disperse, x_disperse, type_id, flux_k2D):
    """
    Function to construct templates that consider all possible spectral combinations of stars
    @u_pix :: max pixels in FOV
    @xy_disperse :: row indicies on the grid i.e. the spectrum following the star
    @type_id :: array storing information about the type of star i.e T_eff = 12,000 K is given a type id of 0, so all stars of the same type are given a similar id number
    @flux_k2D :: k band flux for each star type considered
    
    Returns ::
    @perms :: 2darray with all possible permutations of stars for the given population distribution 
    """
    # array generates all possible realizations of stars with spectra
    perms = generateAllCombinations(type_id)
    flux_PSF_2Dmatrix = np.load('Data/10_star_model/flux_PSF_2Dmatrix.npy')
    
    for i in range(len(perms)):
        flux_matrix2D = np.zeros((u_pix, u_pix))
        
        # reorder the flux arr for the given permutation
        flux_k2D_temp = flux_k2D[np.array(perms[i])]
        
        # construct the 2D matrix
        flux_matrix2D = ss.construct2DFluxMatrix(flux_matrix2D, y_disperse, x_disperse, \
                                         flux_k2D_temp, u_pix) 
        
        # add psf and noise
        flux_matrix2D = ss.addNoise(flux_matrix2D + flux_PSF_2Dmatrix, u_pix)
        # save the file
        num_hot_stars = countHotStars(perms)
        np.save('Data/Template_library_10_stars/hot_stars%d/fluxMat_%dstar_%dpermNo.npy'%(num_hot_stars, len(x_disperse), i), flux_matrix2D)
        np.save('Data/Template_library_10_stars/hot_stars%d/perm_arr.npy'%num_hot_stars, perms)    
    return perms


def splitFOVtoGrid(flux_matrix2D, num_splits):
    """
    Function to split the FOV into a grid of x by x pixels to calculate chi-squared
    @flux_matrix2D :: 2D matrix holding the normalized values of fluxes in the FOV
    @num_splits :: reduced dimension of the flux matrix
    Returns::
    @min_statistic :: the statistic used to define the optimal model (minimizing)
    """
    store_vals_arr = []
    
    # split the flux matrix into rows
    split_rows =  np.vsplit(flux_matrix2D, flux_matrix2D.shape[0]/2)

    for i in range(num_splits):
        # split the 'splited flux matrix' further, but this time by columns
        split_cols = np.hsplit(split_rows[i], num_splits)
        
        # store the sum of these vals 
        store_vals_arr.append([np.sum(split_cols[j]) for j in range(num_splits)])        
    return store_vals_arr

def loadTemplates10Stars(perms_arr, num_stars, idx):
    """
    Function to load template files
    @perms_arr :: arr with info about all possible permutations of the input array
    @num_stars :: the number of stars for which we consider all the possible permutations
    @idx :: idx number, and iterator that runs over all the permutations
    
    Returns :: filename of the template
    """
    num_hot_stars = countHotStars(perms_arr)
    return np.load('Data/Template_library_10_stars/hot_stars%d/fluxMat_%dstar_%dpermNo.npy'%(num_hot_stars, num_stars, idx))
    
    
def loadGenerateTemplates(template_dir, save_data_psf_crop_dir, num_stars, hot_stars, params, idx, disperse_range, dispersion_angle, u_pix, limits):
    # load the flux arr that is used to generate 2D templates
    flux_k2D = np.load(template_dir+'%d_stars/%d_hot_stars/flux_%dperm.npy'%(num_stars, hot_stars*num_stars, idx)) 
    x, y, mKs = params[0], params[1], params[2]
    
    # multiplying the normalized flux my the flux of the star 
    flux_k2D = [flux_k2D[i]*(10**(7-0.4*mKs[i])) for i in range(len(mKs))]
    
    # associate the flux spectra to stars
    waves_k = np.load('Data/waves_k.npy')
    x_disperse, y_disperse = pt.plotDispersedStars('', x, y, disperse_range, waves_k, \
                                               dispersion_angle, no_plot=True)
    
    # construct the 2D matrix
    flux_mat = np.zeros((u_pix[1], u_pix[0]))
    flux_mat = ss.construct2DFluxMatrix(flux_mat, y_disperse, x_disperse, flux_k2D, u_pix)    
    return flux_mat

def determineBestFit10Stars(u_pix, num_stars, perms_arr, data_vals_2Darr, num_splits):
    """
    Function to calculate the difference between the 'data image' and 'template image'
    @u_pix :: dimensions of the FOV
    @num_stars :: number of stars in the FOV
    @perms_arr :: arr with info about all possible permutations of the input array
    @data_flux_matrix2D :: slitless image that is considered as the 'data-input'
    Returns ::
    @chi_squared :: ndarray that stores the chi-squared for every permutation  
    @min_idx :: the permutation no. for which the chi-squared in minimum
    @template_flux_matrix2D :: the template that minimizes the chi-squared
    """
    chi_squared = []
    
    for i in range(len(perms_arr)):
        diff_vals = 0
        template_flux_matrix2D = loadTemplates10Stars(perms_arr, num_stars, i)
        
        # reduce the dimensions of the matrix to cal. the minimizing statistic
        temp_vals_2Darr = splitFOVtoGrid(template_flux_matrix2D, num_splits)
        
        # calculate statistic betweek 'data image and template image'
        for row in range(num_splits):
            for col in range(num_splits):
                model = temp_vals_2Darr[row][col]
                data = data_vals_2Darr[row][col]
                diff_vals += (model-data)**2
                
        # cal the final chi-sq statistic for each template and save it in an array
        chi_squared.append(np.sum(diff_vals))        
        
        # shows a progress bar during computations        
        ss.showProgress(i, len(perms_arr))
        
    # find the template with the minimime chi-squared
    min_idx = np.where(chi_squared == np.min(chi_squared))
    template_flux_matrix2D = loadTemplates(perms_arr, num_stars, min_idx[0][0])
    
    # print result       
    print('\n'+r' The best-fitting permutation is %d with chi-squared = %.2f'%(min_idx[0][0], np.min(chi_squared)))
    return chi_squared, min_idx, template_flux_matrix2D