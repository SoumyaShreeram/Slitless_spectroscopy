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

"""

### 1. Functions for reading in the catalog

"""

def readCatalogFile(filename):
    """
    Function to read the magnitude and positions of the stars from the catalog file
    """
    hdu_list = fits.open(filename)
    # Ks band magnitude
    mag_Ks = hdu_list[1].data['Ksmag']
    mag_H = hdu_list[1].data['Hmag']

    # position of stars in the sky
    ra_Ks = hdu_list[1].data['RAKsdeg']
    de_Ks = hdu_list[1].data['DEKsdeg']

    # errors on magnitude and positions
    e_mag_Ks = hdu_list[1].data['e_Ksmag']
    e_ra_Ks = hdu_list[1].data['e_RAKsdeg']
    e_de_Ks = hdu_list[1].data['e_DEKsdeg']
    
    # close file
    hdu_list.close()
    
    errors = [e_mag_Ks, e_ra_Ks, e_de_Ks]
    return mag_Ks, ra_Ks, de_Ks, errors, mag_H

def decideNumHotStars(hot_stars):
    """
    Function to produce an array holding information about the stellar distribution
    @hot_stars :: number between 0.1-0.9 to define the percent of hot stars in the population
    """
    #TODO: try, catch to correct numbers not between 0.1 and 0.9
    # t_eff_arr = [12000, 11800, 10000, 8400, 7600, 6900, 5900, 5100, 4700, 3900]    
    stars_divide = [hot_stars, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1-hot_stars, 0.0, 0.0]
    return stars_divide

"""

### 2. Functions for selection of stars

"""

def selectFOV(de_ll, de_ul, de_Ks):
    """
    Function selects stars in the given region of the sky
    @de_ll, de_ul :: declination lower limits (ll) and upper limits (ul)
    @de_Ks :: declination indicies
    """
    de_idx_array = []
    
    # select desired dec coordinates
    for m, de in enumerate(de_Ks):
        if de < de_ll and de > de_ul:
            de_idx_array.append(m)
            
    print('Choosing %.2f percent stars from %d total stars.'%((len(de_idx_array)/len(de_Ks))*100, len(de_Ks)))
    return de_idx_array

def selectRealStars(mag_Ks, ra_Ks, de_Ks):
    """
    Function selects real stars in the FOV
    @de_Ks, ra_Ks, mag_Ks :: declination , right ascension, and magnitude of stars arr
    """
    
    idx = np.where(mag_Ks!=99)
    print("Selecting real stars...")
    return mag_Ks[mag_Ks!=99], ra_Ks[idx], de_Ks[idx]

def cutOffFlux(mag_Ks, ra_Ks, de_Ks, cut_off_ll):
    """
    Function cuts off stars below a certain flux from the bottom
    @de_Ks, ra_Ks, mag_Ks :: declination , right ascension, and magnitude of stars arr
    @cut_off_ll :: cut-off limit on the magnitude (note: mag scale is -ve)
    """
    
    idx = np.where(mag_Ks<cut_off_ll)
    print("Discarding stars with magnitude > %d."%cut_off_ll)
    return mag_Ks[mag_Ks<cut_off_ll], ra_Ks[idx], de_Ks[idx]

def selectMaxStars(mag_Ks, ra_Ks, de_Ks, max_stars):
    """
    Function selects #max_stars randomly within the FOV
    @de_Ks, ra_Ks, mag_Ks :: declination , right ascension, and magnitude of stars arr
    """
    
    idx = ss.generateRandInt(0, len(mag_Ks)-1, max_stars)
    print("Selecting a max of %d stars in the FOV randomly."%max_stars)
    return mag_Ks[idx], ra_Ks[idx], de_Ks[idx], idx

def mapToFOVinPixels(de_Ks, ra_Ks, u_pix):
    """
    Function to map the r.a. and declination positions into pixels 
    @de_Ks, ra_Ks, u_pix :: declination , right ascension, and size of the FOV in pixels
    """
    funcX = interp1d([np.min(np.abs(de_Ks)), np.max(np.abs(de_Ks))], [0, u_pix-1])
    funcY = interp1d([np.min(ra_Ks), np.max(ra_Ks)],[0, u_pix-1]) 
    return funcX(np.abs(de_Ks)), funcY(ra_Ks)

def associateSpectraToStars(waves_k, stars_divide, max_stars, flux_LSF2D, params):
    """
    Function to associate the spectra of a certain type to the star
    @waves_k :: wavelength arr in the k-band
    @stars_divide :: arr that categorieses the #stars
    @max_stars :: max_number of stars in the FOV
    @flux_LSD2D :: flux array containins spectra for different spectral_types
    @params :: arr containing [t_eff, log_g, Fe_H, alpha, spectral_types]
    
    Returns 
    @flux_k2D :: 
    @type_id :: arr keeps track of the different types of stars with labels running from 0 to len(star_divide)
    """
    label_arr = np.arange(len(stars_divide))
    flux_k2D = np.zeros((0, len(waves_k)))
    type_id = []
    
    for idx, num_stars in enumerate(stars_divide):
        if num_stars*max_stars != 0:
            for i in range(int(num_stars*max_stars)):
                # keep track of the type of the star
                type_id.append(label_arr[idx])
                
                # add the spectra for the star into the array
                flux_k2D = np.append(flux_k2D, [flux_LSF2D[idx]], axis=0)
            print('%d stars at Teff = %s K, log g = %s'%(num_stars*max_stars, params[idx][0], params[idx][1]))
            
        # if there are no stars with the chosen spectrum
        else:            
            print('%d stars at Teff = %s K, log g = %s'%(num_stars*max_stars, params[idx][0], params[idx][1]))        
    return flux_k2D, type_id

"""

### 3. Functions for generating a template library and choosing the best template

"""

def generateAllCombinations(type_id):
    "Function generates all possible distinguishable combinations of input array"
    perms = set(permutations(type_id))
    return list(perms)

def constructDataImage(perms, u_pix, flux_k2D, y_disperse, x_disperse):
    """
    Function to construct an arbitary 'data image' that considers a possible spectral combinations of stars
    """
    flux_matrix2D = np.zeros((u_pix, u_pix))
    flux_PSF_2Dmatrix = np.load('Data/flux_PSF_2Dmatrix.npy')
    
    # choosing a random permutation number
    idx = ss.generateRandInt(0, len(perms), 1)
    
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

def constructSpectralTemplates(u_pix, y_disperse, x_disperse, type_id, flux_k2D):
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
    flux_PSF_2Dmatrix = np.load('Data/flux_PSF_2Dmatrix.npy')
    
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
    split_rows =  np.vsplit(flux_matrix2D, num_splits)

    for i in range(num_splits):
        # split the 'splited flux matrix' further, but this time by columns
        split_cols = np.hsplit(split_rows[i], num_splits)
        
        # store the sum of these vals 
        store_vals_arr.append([np.sum(split_cols[j]) for j in range(num_splits)])        
    return store_vals_arr

def loadTemplates(perms_arr, num_stars, idx):
    """
    Function to load template files
    @perms_arr :: arr with info about all possible permutations of the input array
    @num_stars :: the number of stars for which we consider all the possible permutations
    @idx :: idx number, and iterator that runs over all the permutations
    
    Returns :: filename of the template
    """
    num_hot_stars = countHotStars(perms_arr)
    return np.load('Data/Template_library_10_stars/hot_stars%d/fluxMat_%dstar_%dpermNo.npy'%(num_hot_stars, num_stars, idx))
     

def determineBestFit(u_pix, num_stars, perms_arr, data_vals_2Darr, num_splits):
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
        diff_val = 0
        template_flux_matrix2D = loadTemplates(perms_arr, num_stars, i)
        
        # reduce the dimensions of the matrix to cal. the minimizing statistic
        temp_vals_2Darr = splitFOVtoGrid(template_flux_matrix2D, num_splits)
        
        # calculate statistic betweek 'data image and template image'
        for row in range(num_splits):
            for col in range(num_splits):
                m = temp_vals_2Darr[row][col]
                n = data_vals_2Darr[row][col]
                diff_val += (m-n)**2

        # cal the final chi-sq statistic for each template and save it in an array
        chi_squared.append(diff_val)        
        
        # shows a progress bar during computations        
        ss.showProgress(i, len(perms_arr))
        
    # find the template with the minimime chi-squared
    min_idx = np.where(chi_squared == np.min(chi_squared))
    template_flux_matrix2D = loadTemplates(perms_arr, num_stars, min_idx[0][0])
    
    # print result       
    print(r'The best-fitting permutation is %d with chi-squared = %.2f'%(min_idx[0][0], np.min(chi_squared)))
    return chi_squared, min_idx, template_flux_matrix2D