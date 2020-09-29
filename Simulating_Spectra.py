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

# Commented out IPython magic to ensure Python compatibility.
import astropy.io.fits as fits
from scipy.sparse import csr_matrix

import numpy as np
import os

# generate random integer values
from random import seed
from random import randint

# plotting imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

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
