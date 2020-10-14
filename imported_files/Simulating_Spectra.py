# -*- coding: utf-8 -*-
"""01_Simulating_Spectra.ipynb

## 01. Preliminary simulation of spectra for 10 stars using Slitless-Spectroscopy

This notebook uses the spectral library [Phoenix](http://phoenix.astro.physik.uni-goettingen.de/) to study the synthetic spectra from stars of different spectra types. Ten stars of different effective temperature $T_e$ and surface gravity $g$ are chosen, and their spectra is plotted. Furthermore, this notebook attempts to recover the spectral image one would obtain when observing these stars in the slitless spectroscopy mode.

The notebook is divided into the following sections:

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

"""### 1. Functions for defining input parameters
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



"""### 2. Opening and reading the file
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
  filename = os.path.join(data_dir, spectral_file_name)
  
  # open the file and extract flux
  hdu_list = openFile(filename)
  flux = hdu_list[0].data
  return wave_len, flux

def setLabel(ax, xlabel, ylabel, title, xlim, ylim, legend=True):
    """
    Function defining plot properties
    @param ax :: axes to be held
    @param xlabel, ylabel :: labels of the x-y axis
    @param title :: title of the plot
    @param xlim, ylim :: x-y limits for the axis
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if legend:
      ax.legend(loc=(1.04,0))
    ax.grid(True)
    ax.set_title(title, fontsize=18)
    return
  
def labelSpectra(params):
  print(params[0])
  "Function labels the spectra based on the parameters"
  spectral_params = (str(params[0]), str(params[1]))
  label = r'$T_e =%s, \log g=%s$'
  return label%spectral_params


"""### 3. Forming a spectral image
"""

def chooseKband(wave_len_arr2D, k_upper, k_lower):
  """
  Function selects the K-band range of wavelengths [2.1-2.4] micrometers
  @k_upper, k_lower :: the upper and lower limits for K-band wavelengths
  @wave_len_arr2D :: wavelength array covering the whole range of the observed spectrum
  
  Returns :: @waves_k :: chosen wavelength array
  """
  waves_k, idx =  [], []
  wave_len = wave_len_arr2D[0]
  
  for i, l in enumerate(wave_len):
    if l <= k_upper and l >= k_lower:
      waves_k.append(l)
      idx.append(i)
  return waves_k, idx

def fluxKband(flux_arr2D, pos, flux_k2D, idx):
  """
  Function for choosing the flux in the K band
  @flux_arr2D :: flux for all wavelengths
  @pos :: arr with random positions
  @flux_k2D :: 
  """
  for i in range(len(pos)):
    flux_k = flux_arr2D[i][idx]
    flux_k2D = np.append(flux_k2D, [flux_k], axis=0)
  return flux_k2D

def disperseStars(x_pos, y_pos, disperse_range, waves_k,  ax, dispersion_angle):
  """
  Function to disperse the flux coming from a star
  @x_pos, y_pos :: the x and y position of the star  
  @disperse_range :: range of wavelength in pixels chosen for dispersion
  @waves_k :: resolution with which the light is dispersed/spectra is binned into
  @ax :: axes handle for plotting the dispersion
  @direction :: variable sets the orientation of the dispersion
  """
  x_disperse, y_disperse = np.empty((0, len(waves_k))), np.empty((0, len(waves_k)))
  # dispersion range of wavelength
  for i, x in enumerate(x_pos):
    x_d = np.linspace(x, x+disperse_range, len(waves_k))  
    
    # convert degree to radians
    angle = (dispersion_angle*np.pi)/180

    intercept = y_pos[i]-np.tan(angle)*x_d
    y_d = np.tan(angle)*x_d + intercept 
    
    ax.plot(x_d, y_d, 'r.')
    
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
    
    # checks if the spectra along the column are exceeding FOV
    if np.any(col>u_pix):
        col = col[col<u_pix]
        row = row[0:len(col)]
        edge_cutter = len(col)
    
    # checks if the spectra along the row are exceeding FOV
    if np.any(row>u_pix):
        row = row[row<u_pix]
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
    for i in range(len(y_disperse)):
        row = y_disperse[i]
        col = x_disperse[i]        
        data_norm = flux_k2D/np.max(flux_k2D)
        
        # makes sure to not consider contributions outside the FOV
        col, row, edge_cutter = checkIfInsideFOV(col, row, u_pix)  
        
        # csr_matrix from scipy puts together a 2D matrix with the desired info
        temp = csr_matrix((data_norm[i][0:edge_cutter], (row, col)), shape=(u_pix, u_pix)).toarray()

        # add all the contributions from all the stars
        flux_matrix2D = flux_matrix2D + temp
    return flux_matrix2D

"""4.1 Add noise

"""

def plotContour(l_pix, u_pix, flux_matrix2D):
    """
    Function plots a contour map for the dispersion caused by slitless spectroscopy
    @noise_level :: decides the amplitude of noise to add to the flux (in %)
    @u_pix :: number of pixels in the FOV
    @disperse_range :: the length dispersion range for each star

    @Returns :: noise_matrix2D :: 2D noise matrix
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))

    X, Y = np.meshgrid(np.linspace(0, u_pix, u_pix), np.linspace(0, u_pix, u_pix))
    plot = ax.contourf(X, Y, flux_matrix2D, cmap='YlOrRd')

    # labeling and setting colorbar
    setLabel(ax, 'x-axis position', 'y-axis position', '', [l_pix, u_pix], [l_pix, u_pix], legend=False)
    cbar = plt.colorbar(plot, aspect=10);
    return

def addNoise(noise_level, u_pix):
    """Function adds noise to the 2D array 
    @noise_level ::decis the amplitude
    @u_pix :: number of pixels in FOV
    @disperse_range :: the length of dispersion fr each star

    @Returns :: noise_matrix2D 2Darray of flux with noise
    """
    shape=(u_pix, u_pix)
    noise_matrix2D = np.random.normal(0, (noise_level*1)/100, size=shape)
    return noise_matrix2D

"""4.2 Add LSF, PSF

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

def redispersedXarr(i_start, i_end, x_pos, x_disperse, disperse_range):
    """
    Function to redefine the x-axis dispersion after adding the LSF
    @i_start, i_end :: start and end indcies around which the x-axis is stretched
    @x_pos :: positions of the stars in the 2x2 grid
    @x_disperse :: shape=(#stars, len(waves_k)) is the arr that holds info about the spectra for all stars
    @disperse_range :: size of the dispersion for each star
    """
    x_disperse_new = np.empty((0, len(x_disperse[0])))
    for idx, x in enumerate(x_pos):
        # define the start and stop indicies
        start = x_disperse[idx][i_start]-disperse_range/20
        stop = x_disperse[idx][i_end]+disperse_range/20
        
        # check that the start point is not negative (<0)
        if start < 0:
            start = 0
        
        # create the 1D array for every star and store all info in the 2D array
        x_arr1D = np.linspace(start, stop, len(x_disperse[0]))
        x_disperse_new = np.append(x_disperse_new, [x_arr1D], axis=0)
    return x_disperse_new

def addLSF(xy_pos, xy_disperse, sigma_xy, disperse_range, factor_widen):
    """
    Function adds smearing to the spectra in the x (LSF) and y (PSF) directions
    @xy_pos :: x- or y-position of the star on the grid
    @xy_disperse :: row indicies on the grid i.e. the spectrum following the star
    @sigma_xy :: 1 sigma deviations from the mean 
    @dispersion_range :: range over which the LSF affects the flux
    @factor_widen :: factor by which the LSF/PSF widens
    
    @Returns :: flux_LSF2D :: 2D array holding all the flux values
    """
    # define the new x-direction dispersed array (boundaries widenned)
    flux_SF2D = np.empty((0, len(xy_disperse[0])))
    
    # add lsf to every point of the spectrum, for every star
    for idx, xy in enumerate(xy_pos):
        z_lsf = np.zeros(len(xy_disperse[0]))
        
        # for every wavelength/point in the spectrum 
        for i in range(len(xy_disperse[0])):
            start = xy_disperse[idx][i]-disperse_range/factor_widen
            stop = xy_disperse[idx][i]+disperse_range/factor_widen
            xy_temp = np.linspace(start, stop, len(xy_disperse[0]))
        
            # producing a normal distribution at a different mean but with a same sigma_x 
            # TODO: make sigma_x wavelength dependent
            z_temp = stats.norm(xy_disperse[idx][i], sigma_xy)
            z_lsf += z_temp.pdf(xy_temp)
                                    
        # information about the background flux contribution due to LSF for every star is stored
        flux_SF2D = np.append(flux_SF2D, [z_lsf], axis=0)
        # shows a progress bar during computations
        showProgress(idx, len(xy_pos))
    return flux_SF2D


def addPSF(xy_pos, xy_disperse, sigma_xy, disperse_range, factor_widen):
    """
    Function adds smearing to the spectra in the x (LSF) and y (PSF) directions
    @xy_pos :: x- or y-position of the star on the grid
    @xy_disperse :: row indicies on the grid i.e. the spectrum following the star
    @sigma_xy :: 1 sigma deviations from the mean 
    @dispersion_range :: range over which the LSF affects the flux
    @factor_widen :: factor by which the LSF/PSF widens
    
    @Returns :: flux_LSF2D :: 2D array holding all the flux values
    """
    # define the new x-direction dispersed array (boundaries widenned)
    flux_PSF2D = np.empty((0, len(xy_disperse[0])))
    flux_PSF3D = np.empty((0, len(xy_disperse[0]), len(xy_disperse[0])))
    
    # add lsf to every point of the spectrum, for every star
    for idx, xy in enumerate(xy_pos):
        z_lsf = np.zeros(len(xy_disperse[0]))
        
        # for every wavelength/point in the spectrum 
        for i in range(len(xy_disperse[0])):
            start = xy_disperse[idx][i]-disperse_range/factor_widen
            stop = xy_disperse[idx][i]+disperse_range/factor_widen
            xy_temp = np.linspace(start, stop, len(xy_disperse[0]))
        
            # producing a normal distribution at a different mean but with a same sigma_x 
            # TODO: make sigma_x wavelength dependent
            z_temp = stats.norm(xy_disperse[idx][i], sigma_xy)
            z_lsf = z_temp.pdf(xy_temp)
            
            # information about the background flux contribution due to point for 1 star is stored
            flux_PSF2D = np.append(flux_PSF2D, [z_lsf], axis=0)
        
        flux_PSF3D = np.append(flux_PSF3D, [flux_PSF2D], axis=0)
        # shows a progress bar during computations
        showProgress(i, len(xy_disperse[0]))
            
    return flux_PSF3D
