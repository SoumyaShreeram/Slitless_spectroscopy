# -*- coding: utf-8 -*-
"""

## 01. Preliminary simulation of spectra for 10 stars using Slitless-Spectroscopy

This python file contains most of the functions used in 01_Simulating_Spectra.ipynb. The data used here comes from the spectral library [Phoenix](http://phoenix.astro.physik.uni-goettingen.de/). The purpose of this file is to study the synthetic spectra from stars of different spectra types. Ten stars of different effective temperature $T_e$ and surface gravity $g$ are chosen, and their spectra is plotted. Furthermore, this file attempts to recover the spectral image one would obtain when observing these stars in the slitless spectroscopy mode.

The functions in this file are divided into the following sections:

1. Defining the input parameters
2. Spectral parameters
3. Opening and reading the file
4. Plotting the spectra
5. Forming a spectral image

**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 29th September 2020

## 1. Imports
"""

import astropy.units as u
import astropy.io.fits as fits

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

"""

### 1. Functions for defining input parameters

"""

def generateRandInt(lower_limit, upper_limit, num_points):
    """
    Function to generate random integers
    @lower_limit, upper_limit :: bounds between which the integers are generated
    @num_points :: number of random points generated

    @Returns :: random_arr :: array on len(num_poits) containing random integers
    """
    random_arr = []
    for _ in range(num_points):
        value = randint(lower_limit, upper_limit)
        random_arr.append(value)
    return random_arr
  
def defineSpectralType(HiRes=True, MedRes=False):
    """
    Function decides which spectra to use: High-Resolution or Mid-Resolution
    @HiRes :: boolean selects spectra with resolution 5*e5 in optical and NIR
    @MidRes :: boolean selects spectra with resolution 1*e5 in IR
    """
    spectral_types_arr = ['HiRes','MedRes']
    if HiRes:
        spectral_type = spectral_types_arr[0]
    else:
        spectral_type = spectral_types_arr[1]
    return spectral_type

def starPositions(l_pix, u_pix, num_stars, generate_new_pos, filename):
    """
    Function defines the postions of stars in FOV
    @l_pix, u_pix :: bounds for the FOV 
    @generate_new_pos :: boolean decides wether to create new star positions or use old ones
    @filename :: if one decides to use old positions, provide the function with the filename where the positions anre stored
    """
    if generate_new_pos:
        x_pos = generateRandInt(l_pix, u_pix, num_stars) 
        y_pos = generateRandInt(l_pix, u_pix, num_stars)
    else:
        pos_arr = np.load(filename)
        x_pos, y_pos = pos_arr[0], pos_arr[1]
    return x_pos, y_pos

def defineDispersionRange(r, lambda0, band_width):
    """
    Function to find the dispersion range of the spectra
    @r :: resolution of the telescope
    @lambda0 :: equivalant wavelength of the selected band
    @band_width :: wavelengths over which the spectra is obtained
    """
    return (r/lambda0)*band_width
"""

### 2. Opening and reading the file

"""

def defineSpectralFilename(params):
    """
    Function to put together the filename based on the spectral parameters
    @params:: arr containing [t_eff, log_g, Fe_H, alpha, spectral_types]

    @Returns filename :: fits filename for the given parameters
    """
    if params[3] == 0:
        # names the file based on the length of T_eff
        if len(str(params[0])) == 4:
            filename = 'lte0'+str(params[0])+'-'+str(params[1])+'0-'+str(params[2])+'.PHOENIX-ACES-AGSS-COND-2011-'+ params[4]+'.fits'
        else:
            filename = 'lte'+str(params[0])+'-'+str(params[1])+'0-'+str(params[2])+'.PHOENIX-ACES-AGSS-COND-2011-'+ params[4]+'.fits'

    # if alpha is not equal to 0
    else:
        if len(str(params[0])) == 4:
            filename = 'lte0'+str(params[0])+'-'+str(params[1])+'-'+str(params[2])+'.Alpha='+str(params[3])+'.PHOENIX-ACES-AGSS-COND-2011-'+ params[4]+'.fits'
        else:
            filename = 'lte'+str(params[0])+'-'+str(params[1])+'-'+str(params[2])+'.Alpha='+str(params[3])+'.PHOENIX-ACES-AGSS-COND-2011-'+ params[4]+'.fits'
    return filename

def openFile(filename):
    "Function to open the fits file"
    hdu_list = fits.open(filename)
    return hdu_list

def extractWavelen(data_dir, wave_filename):
    "Function finds the length of the wavelength array"
    # open wavelength file and extract data
    wave_array = os.path.join(data_dir, wave_filename)
    hdu_wave = openFile(wave_array)
    wave_len = hdu_wave[0].data
    return wave_len, len(wave_len)

def readFile(data_dir, wave_filename, params):
    """
    Function to define the filename, open the file, and print out some basic info
    Inputs:
    @data_dir :: directory that stores the fits files
    @wave_arr :: wavelength array for all the spectra
    @params :: arr containing [t_eff, log_g, Fe_H, alpha, spectral_types]
    """
    # open wavelength file and extract data
    wave_len, _ = extractWavelen(data_dir, wave_filename)

    # define the spectral file name 
    spectral_file_name = defineSpectralFilename(params)
    filename = os.path.join(data_dir, 'spectra', spectral_file_name)

    # open the file and extract flux
    hdu_list = openFile(filename)
    flux = hdu_list[0].data
    return wave_len, flux


"""

### 3. Forming a spectral image

"""

def resamplingSpectra(arr_x, values, disperse_range, statistic_type):
    """
    Function for resampling the synthetic spectra in the K band as per resolution
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    @arr_x :: array that is binned
    @values :: values on which the statistic will be computed (same shape as arr_x).
    """
    bins = np.linspace(0, 1, disperse_range)
    bin_means,_,_ = stats.binned_statistic(arr_x, values, statistic=statistic_type, bins=disperse_range)
    return bin_means

def chooseKband(wave_len_arr2D, k_upper, k_lower, disperse_range):
    """
    Function selects the K-band range of wavelengths [2.1-2.4] micrometers
    @k_upper, k_lower :: the upper and lower limits for K-band wavelengths
    @wave_len_arr2D :: wavelength array covering the whole range of the observed spectrum
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    
    Returns :: @waves_k :: chosen wavelength array in the k-band
    @waves_k_binned :: chosen wavelength array in the k-band binned to the size of disperse_range
    @idx :: index arr of chosen wavelengths that is useful to extract corresponding flux
    """
    waves_k, idx = [], []
    wave_len = wave_len_arr2D[0]

    for i, l in enumerate(wave_len):
        if l <= k_upper and l >= k_lower:
            waves_k.append(l/10000)
            idx.append(i)
    
    # rebinning the data based on the dispersion range inputted
    waves_k_binned = resamplingSpectra(waves_k, waves_k, disperse_range, 'mean')
    return waves_k, idx, waves_k_binned

def fluxKband(waves_k, flux_arr2D, pos, flux_k2D, idx, disperse_range, statistic_type):
    """
    Function for choosing the flux in the K band
    @flux_arr2D :: flux for all wavelengths
    @pos :: arr with random positions
    @flux_k2D :: 2Darrays holding info about the flux of all the stars in the k-band
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    """
    for i in range(len(pos)):
        # choose k band values
        flux_k = flux_arr2D[i][idx]
        
        # resample/rebin the values based of input resolution (disperse_range)
        flux_k = resamplingSpectra(waves_k, flux_k, disperse_range, statistic_type)
        
        # append these values to the 2D arr
        flux_k2D = np.append(flux_k2D, [flux_k], axis=0)
    return flux_k2D

def disperseStars(x_pos, y_pos, disperse_range, waves_k,  ax, dispersion_angle, no_plot):
    """
    Function to disperse the flux coming from a star
    @x_pos, y_pos :: the x and y position of the star  
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    @waves_k :: resolution with which the light is dispersed/spectra is binned into
    @ax :: axes handle for plotting the dispersion
    @dispersion_angle :: variable sets the orientation of the dispersion
    """
    x_disperse, y_disperse = np.zeros((0, len(waves_k))), np.zeros((0, len(waves_k)))
    # dispersion range of wavelength
    for i, x in enumerate(x_pos):
        x_d = np.linspace(x, x+disperse_range, len(waves_k))  

        # convert degree to radians
        angle = (dispersion_angle*np.pi)/180

        intercept = y_pos[i]-np.tan(angle)*x_d
        y_d = np.tan(angle)*x_d + intercept
        
        if not no_plot:
            ax.plot(x_d, y_d, '.', color='#d8b6fa')

        # save the values
        x_disperse = np.append(x_disperse, [x_d], axis=0)
        y_disperse = np.append(y_disperse, [y_d], axis=0)
    return x_disperse, y_disperse 

def checkIfInsideFOV(col, row, u_pix):
    """ 
    Function makes sure to NOT consider contributions outside the FOV
    """
    # if there are no issues, leave this as it is
    edge_cutter = len(col)
    
    # if FOV is a symmetric square
    if isinstance(u_pix, (int, float)):
        # checks if the spectra along the column are exceeding FOV
        if np.any(col>u_pix-1):
            col = col[col<u_pix]
            row = row[0:len(col)]
            edge_cutter = len(col)

        # checks if the spectra along the row are exceeding FOV
        if np.any(row>u_pix-1):
            row = row[row<u_pix]
            col = col[0:len(row)]
            edge_cutter = len(row)
            
    # if FOV is a rectangle
    if isinstance(u_pix, (list, tuple, np.ndarray)):
        if np.any(col>u_pix[0]-1):
            col = col[col<u_pix[0]]
            row = row[0:len(col)]
            edge_cutter = len(col)

        # checks if the spectra along the row are exceeding FOV
        if np.any(row>u_pix[1]-1):
            row = row[row<u_pix[1]]
            col = col[0:len(row)]
            edge_cutter = len(row)
            
    return col, row, edge_cutter

def construct2DFluxMatrix(flux_matrix2D, y_disperse, x_disperse, flux_k2D, u_pix):
    """
    Function for constructing a 2D flux matrix that is used for plotting a spectral image in slitless mode
    @flux_matrix2D :: 2D matrix of dimensions = (size of one grid/pointing, size of one grid/pointing)
    @y_disperse, x_disperse :: 2Darrays holding info about the dispersion due to each star on an x-y grid
    @flux_k2D :: 2Darrays holding info about the flux of all the stars in the k-band
    @u_pix, disperse_range :: max number of pixels, size of the dispersion for each star
    
    @Returns :: flux_matrix2D :: 2D matrix filled with values
    """
    # if FOV is a symmetric square
    if isinstance(u_pix, (int, float)):
        shape = (u_pix, u_pix)
    if isinstance(u_pix, (list, tuple, np.ndarray)): # if FOV is a rectangle
        shape = (u_pix[1], u_pix[0]) 
        
    for i in range(len(y_disperse)):
        row = y_disperse[i]
        col = x_disperse[i]   
                
        # makes sure to not consider contributions outside the FOV
        col, row, edge_cutter = checkIfInsideFOV(col, row, u_pix)  
        
        # csr_matrix from scipy puts together a 2D matrix with the desired info
        temp = csr_matrix((flux_k2D[i][0:edge_cutter], (row, col)), shape=shape).toarray()
        flux_matrix2D = flux_matrix2D+temp
    return flux_matrix2D

"""

4.1 Add noise

"""
def addNoise(flux_2Dmat, u_pix):
    """Function adds random noise to the 2D array 
    @noise_level :: lambda parameter of the Poisson distribution
    @u_pix :: number of pixels in FOV
    @disperse_range :: the length of dispersion fr each star

    @Returns :: noise_matrix2D 2Darray of flux with noise
    """
    if isinstance(u_pix, (int, float)):
        shape=(u_pix, u_pix)
        noise_matrix2D = np.random.poisson(lam=flux_2Dmat, size=shape)
        
    if isinstance(u_pix, (list, tuple, np.ndarray)):
        shape=(u_pix[0], u_pix[1])
        noise_matrix2D = np.random.poisson(lam=flux_2Dmat, size=shape)
    return noise_matrix2D
"""

4.2 Add LSF, PSF

"""

def showProgress(idx, n):
    """
    Function prints the progress bar for a running function
    @param idx :: iterating index
    @param n :: total number of iterating variables/ total length
    """
    j = (idx+1)/n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
    sleep(0.25)
    return

def addLSF(xy_pos, flux_k2D, sigma_LSF, waves_k):
    """
    Function adds smearing to the spectra in the x (LSF) and y (PSF) directions
    @xy_pos :: x- or y-position of the star on the grid
    @flux_k2D :: ndarray containing info about the flux for all the stars
    @sigma_LSF :: 1 sigma deviations for the Gaussian
    @waves_k :: wavelength array in the k-band
    
    @Returns :: flux_LSF2D :: 2D array holding all the flux values
    """
    # define the new x-direction dispersed array (boundaries widenned)
    flux_SF2D = np.zeros((0, len(waves_k)))
    
    for idx, xy in enumerate(xy_pos):        
        # flux for a star in terms of the Spectrum1D object of astropy
        flux_star = np.array(flux_k2D[idx])*u.erg/(u.s*u.cm*u.cm)
        spec = Spectrum1D(spectral_axis=waves_k*u.micron, flux=flux_star)
        
        # smoothing the spectrum
        spec_smoothed = gaussian_smooth(spec, sigma_LSF)
        
        flux_SF2D = np.append(flux_SF2D, [spec_smoothed.flux], axis=0)
        
        # shows a progress bar during computations
        showProgress(idx, len(xy_pos))                       
    return flux_SF2D

def checkNegativeVals(val):
    "Function to check that the value is not negative (<0)"
    if val < 0:
        val = 0
    return val

def addPSF(xy_pos, xy_disperse, sigma_xy, disperse_range, factor_widen, norm):
    """
    Function adds smearing to the spectra in the x (LSF) and y (PSF) directions
    @xy_pos :: x- or y-position of the star on the grid
    @xy_disperse :: row indicies on the grid i.e. the spectrum following the star
    @sigma_xy :: 1 sigma deviations from the mean 
    @dispersion_range :: range over which the LSF affects the flux
    @factor_widen :: factor by which the LSF/PSF widens
    
    @Returns :: flux_LSF2D :: 2D array holding all the flux values
    """
    print('Adding PSF...\n')
    spread_over_pix = int(2*disperse_range/factor_widen)
    
    # define the new flux and position arrays to hold new info   
    flux_PSF3D = np.zeros((0, len(xy_disperse[0]), spread_over_pix))
    xy_temp3D = np.zeros((0, len(xy_disperse[0]), spread_over_pix))
    
    # add psf to every point of the spectrum, for every star
    for idx, xy in enumerate(xy_pos):
        # arrays to store info about the flux and postions on the FOV for EACH star
        flux_PSF2D = np.zeros((0, spread_over_pix))
        xy_temp2D = np.zeros((0, spread_over_pix))
        z_psf = np.zeros(spread_over_pix)
        
        # for every wavelength/point in the spectrum 
        for i in range(len(xy_disperse[0])):
            start = xy_disperse[idx][i]-int(spread_over_pix/2)
            stop = xy_disperse[idx][i]+int(spread_over_pix/2)
            start = checkNegativeVals(start)
            xy_temp = np.linspace(start, stop, spread_over_pix)
        
            # producing a normal distribution at a different mean but with a same sigma_x 
            z_temp = stats.norm(xy_disperse[idx][i], sigma_xy)
            z_psf = norm*z_temp.pdf(xy_temp)
            
            # information about the background flux contribution due to point for 1 star is stored
            flux_PSF2D = np.append(flux_PSF2D, [z_psf], axis=0)
            xy_temp2D = np.append(xy_temp2D, [xy_temp], axis=0)
        
        flux_PSF3D = np.append(flux_PSF3D, [flux_PSF2D], axis=0)
        xy_temp3D = np.append(xy_temp3D, [xy_temp2D], axis=0)
        
        # shows a progress bar during computations        
        showProgress(idx, len(xy_pos))
    return flux_PSF3D, xy_temp3D

def constructFluxMatrixPSF(x_pos, x_disperse, y_dispersePSF, flux_k2D, u_pix):
    """
    Function for constructing a 2D flux matrix that is used for plotting a spectral image in slitless mode
    @x_pos :: x- or y-position of the star on the grid
    @xy_disperse :: row indicies on the grid i.e. the spectrum following the star
    @flux_k2D :: ndarray containing info about the flux for all the stars
    @u_pix :: number of pixels in FOV
    
    @Returns :: flux_LSF2D :: 2D array holding all the flux values
    """
    print('\n\nCreating 2D flux matrix...\n')
    flux_matrix2D = np.zeros((u_pix, u_pix))
    for i in range(len(x_pos)):        
        # for every wavelength/point in the spectrum 
        for idx in range(len(x_disperse[0])):
            row = y_dispersePSF[i][idx]
            col = x_disperse[i][idx]*np.ones(len(row))

            # makes sure to not consider contributions outside the FOV
            col, row, edge_cutter = checkIfInsideFOV(col, row, u_pix)  

            # csr_matrix from scipy puts together a 2D matrix with the desired info
            flux_matrix2D += csr_matrix((flux_k2D[i][idx][0:edge_cutter], (row, col)), shape=(u_pix, u_pix)).toarray()
            
        # shows a progress bar during computations        
        showProgress(i, len(x_pos))        
    return flux_matrix2D

def printForgroundPopulation(mag, max_stars):
    """
    Function prints the number of stars that belong to the foreground population
    """
    fore_stars = len(np.where(mag<1.1)[0])
    return print(r'Forground population of stars in sample (H-K_s < 1.1): %d stars'%fore_stars)
