# -*- coding: utf-8 -*-
"""Plotting.py

## Plotting functions for simulating stars using Slitless-Spectroscopy

This python file contains all the functions used for plotting graphs and density maps across the various notebooks.

**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 8th October 2020

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

import matplotlib

# for manupilating spectra
from specutils.manipulation import (box_smooth, gaussian_smooth, trapezoid_smooth)
from specutils import Spectrum1D

# personal file imports
import Simulating_Spectra as ss

"""### 1. Functions for labeling plots
"""

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
    
    if xlim != 'default':
        ax.set_xlim(xlim)
    if ylim != 'default':
        ax.set_ylim(ylim)
    
    if legend:
      ax.legend(loc=(1.04,0))
    ax.grid(True)
    ax.set_title(title, fontsize=18)
    return
  
def labelSpectra(params):
    "Function labels the spectra based on the parameters"
    spectral_params = (str(params[0]), str(params[1]))
    label = r'$T_e =%s, \log g=%s$'
    label = label%spectral_params
    return label

def shortenXYaxisTicks(ax):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return

"""### 2. Functions for plotting other things
"""

def plotDispersedStars(x_pos, y_pos, l_pix, u_pix, disperse_range, waves_k, dispersion_angle):
    """
    Function plots a contour map for the dispersion caused by slitless spectroscopy
    @noise_level :: decides the amplitude of noise to add to the flux (in %)
    @u_pix :: number of pixels in the FOV
    @disperse_range :: the length dispersion range for each star

    @Returns :: noise_matrix2D :: 2D noise matrix
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    

    # dispersing them in the k-band
    x_disperse, y_disperse = ss.disperseStars(x_pos, y_pos, disperse_range, waves_k, ax, \
                                              dispersion_angle)
    # plotting the stars
    ax.plot(x_pos, y_pos, ".", color= '#ebdf09', alpha=0.9, marker="*", markersize=10)
    setLabel(ax, 'x-axis position', 'y-axis position', '', [l_pix, u_pix], \
                [l_pix, u_pix], legend=False)
    return x_disperse, y_disperse

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
    plot = ax.contourf(X, Y, flux_matrix2D, cmap='YlGnBu')

    # labeling and setting colorbar
    setLabel(ax, 'x-axis position', 'y-axis position', '', [l_pix, u_pix], [l_pix, u_pix], legend=False)
    cbar = plt.colorbar(plot, aspect=10);
    return

def plotDifferentGaussianLSFs(ax, flux_k2D, waves_k, sigma_LSF_arr, colors, xlim, ylim):
    """
    Function to access the right choice of sigma for Gaussian smoothing of the spectrum
    @param ax :: axes to be held
    @flux_k2D :: ndarray containing info about the flux for all the stars
    @waves_k :: wavelength array in the k-band
    @sigma_LSF_arr :: 1darray containing trial values for the sigma for Gaussian smoothing
    @colors :: arr with different colors to show in the plot
    @xlim, ylim :: useful argument in case one wants to zoom into a particlular region
    """
    # example spectrum to assess the effects of smoothing
    flux_star = np.array(flux_k2D[0])*u.erg/(u.s*u.cm*u.cm)
    spec1 = Spectrum1D(spectral_axis=waves_k*u.micron, flux=flux_star)
    
    ax.plot(spec1.spectral_axis, spec1.flux, 'k', label='No LSF')
    
    for i, sigma_LSF in enumerate(sigma_LSF_arr):
        spec_smoothed = gaussian_smooth(spec1, sigma_LSF)

        ax.plot(spec_smoothed.spectral_axis, spec_smoothed.flux, color=colors[i], \
                linestyle='--', label=r'$\sigma$=%d'%sigma_LSF, linewidth=2)
        
        # keep title for first plot
        if xlim == 'default':
            setLabel(ax, '', r'Flux (erg/cm$^2$/s)', \
                     r'Testing different $\sigma$ for Gaussian Smoothing (LSF)', \
                        xlim, ylim, legend=True)
        # Ignore title for the zoomed in plot
        else:
            setLabel(ax, r'$\lambda (\mu$m)', r'Flux (erg/cm$^2$/s)', '', xlim, ylim, legend=False)

    ax.set_yscale('log')
    return 

def plotMagDiffs(mag_H, cut_off_ll):
    mag_H = mag_H[mag_H<cut_off_ll]    

    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    ax.plot(mag_H[idx]-mag_Ks, mag_Ks, 'g.')
    #plot vertical line signifying the cut-off
    ax.axvline(forground_cutoff, np.min(mag_Ks), np.max(mag_Ks), color='k')
    setLabel(ax, r'$K_s$-H (mag)', r'$K_s$ (mag)', '', xlim, ylim, legend=False)
    return

def plotLSFaddedSpectra(ax, num_spectra, waves_k, flux_LSF2D, pal, params_arr2D, title):
    for i in range(num_spectra):
        ax.plot(waves_k, flux_LSF2D[i], color=pal[i], label=labelSpectra(params_arr2D[i]))
    
    setLabel(ax, r'$\lambda [\mu m]$', r'Flux [erg/s/$cm^2$]', \
            title+ r'K-band spectra with LSF', 'default',\
            'default', legend=True)       
    return