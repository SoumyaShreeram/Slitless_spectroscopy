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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import matplotlib
from mpl_toolkits import axes_grid1

# personal file imports
import Simulating_Spectra as ss
import Fitting_and_pdfs as fap
import Slitless_spec_forward_modelling as ssfm

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

def plotDispersedStars(ax, x_pos, y_pos, disperse_range, waves_k, dispersion_angle, no_plot):
    """
    Function plots a contour map for the dispersion caused by slitless spectroscopy
    @noise_level :: decides the amplitude of noise to add to the flux (in %)
    @u_pix :: number of pixels in the FOV
    @disperse_range :: the length dispersion range for each star
    @dispersion_angle :: the angle of dispersion
    @no_plot :: boolean to decide wether or not to plot the figure of the dispersed stars

    @Returns :: noise_matrix2D :: 2D noise matrix
    """
    # dispersing them in the k-band
    x_disperse, y_disperse = ss.disperseStars(x_pos, y_pos, disperse_range, waves_k, ax, \
                                              dispersion_angle, no_plot)
    if not no_plot:
        # plotting the stars
        ax.plot(x_pos, y_pos, ".", color= '#ebdf09', alpha=0.9, marker="*", markersize=10)
        setLabel(ax, 'x-axis position', 'y-axis position', '', 'default', \
                    'default', legend=False)
    return x_disperse, y_disperse


def plotContour(u_pix, flux_matrix2D):
    """
    Function plots a contour map for the dispersion caused by slitless spectroscopy
    @noise_level :: decides the amplitude of noise to add to the flux (in %)
    @u_pix :: number of pixels in the FOV
    @disperse_range :: the length dispersion range for each star

    @Returns :: noise_matrix2D :: 2D noise matrix
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    if isinstance(u_pix, (int, float)):
        X, Y = np.meshgrid(np.linspace(0, u_pix, u_pix), np.linspace(0, u_pix, u_pix))
    if isinstance(u_pix, (list, tuple, np.ndarray)): # if FOV is a rectangle
        X, Y = np.meshgrid(np.linspace(0, u_pix[0], u_pix[0]), np.linspace(0, u_pix[1], u_pix[1]))
    
    plot = ax.contourf(X, Y, flux_matrix2D, cmap='YlGnBu')

    # labeling and setting colorbar
    setLabel(ax, 'x-axis position', 'y-axis position', '', 'default', 'default', legend=False)
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

def plotMagDiffs(mag_H, mag_Ks, forground_star_cut, max_stars):
    """
    Function to plot the H-Ks magnitudes for estimating foreground population
    @mag_H :: H-band mag of stars
    @cut_off_ll :: lower limit on which we are cutting off stars
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    ax.plot(mag_H-mag_Ks, mag_Ks, 'g.', markersize=2.5)
    
    #plot vertical line signifying the cut-off
    ax.axvline(forground_star_cut, color='k')
    
    ax.invert_yaxis()    
    setLabel(ax, r'H-$K_s$ (mag)', r'$K_s$ (mag)', '', [np.min(mag_H-mag_Ks), np.max(mag_H-mag_Ks)], [np.max(mag_Ks), np.min(mag_Ks)], legend=False)
    ax.grid(False)
    
    # print foreground population
    ss.printForgroundPopulation(mag_H-mag_Ks, max_stars)
    return

def plotLSFaddedSpectra(ax, num_spectra, waves_k, flux_LSF2D, pal, params_arr2D, title):
    """
    Function plots the spectra after the application of an LSF
    @ax :: holds the axis
    @num_spectra :: number of spectra (and corresponding stars) there are in the FOV
    @waves_k :: wavelength array chosen for the k-band
    @flux_LSF2D :: flux array that contains all the spectra for #stars with LSF applied
    @pal :: shortform for palette, and contains the range of colors considered for plotting
    @params_arr2D :: arr containing [t_eff, log_g, Fe_H, alpha, spectral_types] for each star
    @title :: title to the plot
    """
    for i in range(num_spectra):
        ax.plot(waves_k, flux_LSF2D[i], color=pal[i], label=labelSpectra(params_arr2D[i]))
    
    setLabel(ax, r'$\lambda [\mu m]$', r'Flux [erg/s/$cm^2$]', \
            title+ r' K-band spectra with LSF', 'default',\
            'default', legend=True)       
    return

def plotChiSquare(perms_arr, chi_squared_arr):
    """
    Function plots the chi-squared vs permutation number plot
    @chi_squared_arr :: [chi_squared, min_idx, template_flux_matrix2D] ` contains three quantities
    @perms_arr :: arr containing all the possible permuations for a given configuration
    """
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    
    ax.plot(np.arange(len(perms_arr)), chi_squared_arr[0], 'k--')
    setLabel(ax, 'Permutation number', r'$\chi^2$', '', 'default', \
                    'default', legend=False)
    print('The best fitting permutation has the following spectral types:\n ', perms_arr[chi_squared_arr[1][0][0]]) 
    return

def plotChiSquredDistribution(chi_square3D, hot_stars_arr):
    """
    Function plots the chi-squared vs percent of hot stars
    @chi_squared_arr :: [chi_squared, min_idx, template_flux_matrix2D] ` contains three quantities
    @perms_arr :: arr containing all the possible permuations for a given configuration
    """
    min_chi_idx = [np.min(chi_square3D[i][1][0][0]) for i in range(len(hot_stars_arr))]
    min_chi_sq = [chi_square3D[i][0][min_chi_idx[i]] for i in range(len(hot_stars_arr))]
    
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    ax.plot(hot_stars_arr*100, min_chi_sq, 'k--', marker='*', markerfacecolor='r')
        
    setLabel(ax, '% of hot stars in FOV', r'min[$\chi^2$] among all permutations', '', 'default', \
                    'default', legend=False)
    return

def plotForegroundGCstars(foreground_stars, gc_stars):
    """
    Function to plot the foreground vs gc star distribution as a function of the grid number    
    """
    fig, ax = plt.subplots(2,1,figsize=(9,18))
    
    # plotting both the foreground and gc stars
    ax[0].plot(np.arange(len(foreground_stars)), foreground_stars, '.', color= 'k', alpha=0.9, marker="*", markersize=10, label='Foreground stars')
    ax[0].plot(np.arange(len(foreground_stars)), gc_stars,'.', color= 'r', alpha=0.9, marker="*", markersize=10, label='GC stars')
    setLabel(ax[0], 'Grid or pointing number', 'Number of stars', '', [0, len(foreground_stars)], 'default', legend=True)
    ax[0].grid(False)
    ax[0].legend(loc='upper right')
    
    # plotting only the foreground stars
    ax[1].plot(np.arange(len(foreground_stars)), foreground_stars, '.', color= 'purple', alpha=0.9, marker="1", markersize=10, label='Foreground stars')
    setLabel(ax[1], 'Grid number', '', '', 'default', 'default', legend=True)
    ax[1].legend(loc='upper right')    
    return

"""

Plots for notebook 5

"""

def plotAllNeighbours(x_pos, num_neighbours, neighbours_outside_FOV, total_neighbours):
    """
    Function plots all the neighbours to a given set of stars (defined within the FOV)
    @x_pos :: arr holding the positions of all the stars
    @num_neighbours :: ndarray holding information about the total number of neighbours within and in the total FOV
    @neighbours_outside_FOV :: as the name suggest, star neighbours outside FOV
    @total_neighbours :: total number of neighbours for all stars in the given FOV 
    """
    fig, ax = plt.subplots(2,1,figsize=(9,18))

    # plotting the stars
    ax[0].plot(np.arange(1, len(x_pos)+1), num_neighbours[0], 'o', color="ForestGreen", label='Stars within FOV')
    ax[0].plot(np.arange(1, len(x_pos)+1), neighbours_outside_FOV, 'd', color="red", label='Stars outside FOV')

    setLabel(ax[0], 'Star number', 'Neighbouring stars', '', 'default', 'default', legend=True)

    ax[1].plot(np.arange(1, len(x_pos)+1), total_neighbours, 's', color="DarkBlue")

    setLabel(ax[1], 'Star number', 'Total Neighbouring stars', '', 'default', 'default', legend=False)
    return 

def visualizingNeighbours(idx, x_pos, y_pos, x_FOV, y_FOV, star_neighbours):
    """
    Function to visualize which stars form neighbours to a given star
    @idx :: index to choose a random star
    @x_pos, y_pos :: positions of all the stars in the given FOV
    @star_neighbours :: ndarray holding info about the neighbours' [x_pos, y_pos, mag_Ks, mag_H] for all stars in the given set
    """
    fig, ax = plt.subplots(1,1,figsize=(13,8))
    
    # plotting all the stars in the FOV
    ax.plot(x_pos, y_pos, '.', color="#a3a19b")
        
    # plotting all the neighbours outside FOV to the star of interest
    ax.plot(star_neighbours[1][idx][0], star_neighbours[1][idx][1], 'o', color="#34ebc6", markersize=12, label='All neighbours')
    
    # plotting all the neighbours inside FOV to the star of interest
    ax.plot(star_neighbours[0][idx][0], star_neighbours[0][idx][1], "b*", label='Neighbours within FoV')
    
    # the star of interest itself
    ax.plot(x_FOV[idx], y_FOV[idx], 'r*', markersize=15, label='Target Star')
    
    setLabel(ax, 'x-position (pixels)', 'y-position (pixels)', '', 'default', 'default', legend=True)
    
    # printing the results
    print('Number of neighbouring stars in FOV:', len(star_neighbours[0][idx][0]))
    print('Number of neighbouring stars in effective FOV:', len(star_neighbours[1][idx][0]))
    shortenXYaxisTicks(ax)
    plt.tight_layout()
    return

def plotAllStarsWithNneighbours(star_neighbours, x_pos, y_pos, stars_with_n_neighbours):
    """
    Function to plot all the stars that have 'n' number of neighbours within and outside the FOV
    @
    """
    fig, ax = plt.subplots(1,1,figsize=(16,8))

    for idx in stars_with_n_neighbours[0]:
        # plotting all the neighbours to the star of interest    
        ax.plot(star_neighbours[0][idx][0], star_neighbours[0][idx][1], "bo", markersize=12)
        ax.plot(star_neighbours[1][idx][0], star_neighbours[1][idx][1], "*", color='#54f542', markersize=10)

        # the star of interest itself
        ax.plot(x_pos[idx], y_pos[idx], 'r*', markersize=15)

    setLabel(ax, 'x-axis position', 'y-axis position', '', 'default', \
                    'default', legend=False)
    ax.set_title('Regions with stars containing %d neighbours'%len(star_neighbours[1][idx][0]))
    return 

def showTheRegionOfAnalysis(selected_c_pxls, stars_outside_FOV, x_start, y_start, u_pix):
    """
    Function plots the region that is considered for the analysis
    """
    fig, ax = plt.subplots(1,1,figsize=(10,6))

    # plotting the stars within the FOV
    ax.plot(selected_c_pxls[0], selected_c_pxls[1], ".", color= '#cfbbba',\
            alpha=0.9, marker="*", markersize=0.5, label='Stars')

    # create a Rectangle patch
    rect = Rectangle((x_start, y_start),u_pix,u_pix,linewidth=1,edgecolor='r',\
                     facecolor='None', zorder=10, label='FoV')
    
    # left Rectangle patch    
    rect_left = Rectangle((np.min(stars_outside_FOV[0]), y_start), \
                          x_start-np.min(stars_outside_FOV[0]),u_pix,linewidth=1,edgecolor='r', \
                          facecolor='None', zorder=10, hatch='/')
    
    # right Rectangle patch
    rect_right = Rectangle((x_start+u_pix, y_start), np.max(stars_outside_FOV[0])-x_start-u_pix,u_pix,linewidth=1,edgecolor='r', facecolor='None', zorder=10, hatch='/', label='Outside defined FoV')

    # add the patch to the Axes
    ax.add_patch(rect)
    ax.add_patch(rect_left)
    ax.add_patch(rect_right)
    
    # set labels
    setLabel(ax, 'x-axis position', 'y-axis position', '', [np.min(selected_c_pxls[0]), np.max(selected_c_pxls[0])], [np.min(selected_c_pxls[1]), np.max(selected_c_pxls[1])], legend=True)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    ax.grid(False)
    return

def showGrid(selected_c_pxls, x_all, y_all, mag_H_all, mag_Ks_all, u_pix_arr):
    """
    Function shows how the central catalogue can be divided into grids
    
    """
    fig, ax = plt.subplots(1,1,figsize=(10,7))

    # plotting the stars within the FOV
    ax.plot(selected_c_pxls[0], selected_c_pxls[1], ".", color= '#cfbbba',\
            alpha=0.9, marker="*", markersize=0.5)

    # arrays that set the sizes of the grids
    steps_x = np.arange(np.min(x_all), np.max(x_all), u_pix_arr[0])
    steps_y = np.arange(np.min(y_all), np.max(y_all), u_pix_arr[1])
    
    # looping over every grid of size u_pix
    for i, x in enumerate(steps_x):
        for j, y in enumerate(steps_y):
            limits = ssfm.generateLimits(x, y, u_pix_arr)
            x_grid, y_grid, _, _ = ssfm.selectFOV(limits, x_all, y_all, mag_H_all, mag_Ks_all, print_msg=False)
            
            # create a Rectangle patch
            rect = Rectangle((limits[0], limits[2]), limits[1], limits[3], linewidth=1,edgecolor='r',\
                     facecolor='None', zorder=10)
            
            # add the patch to the Axes
            ax.add_patch(rect)
            
            # add text
            plt.text(limits[0]+150, limits[2]+150,'%d'%(len(steps_y)*i+j))
            
    # set labels
    setLabel(ax, 'x-axis position (pixels)', 'y-axis position (pixels)', '', [np.min(selected_c_pxls[0]), np.max(selected_c_pxls[0])], [np.min(selected_c_pxls[1]), np.max(selected_c_pxls[1])], legend=False)
    plt.tight_layout()
    ax.grid(False)
    return

def plotFluxMatAroundAstar(x_FOV, y_FOV, total_neighbours, disperse_range, width, u_pix_arr, n_stars, region_idx):
    """
    Function to plot a cropped region of the data matrix 
    @x_FOV, y_FOV :: arrays with x-y coordinates of stars in the defined FoV
    @total_neighbours :: arr that holds information about the total number of neighbours for every star in the FoV
    @disperse_range, width :: params to define the region of influence for each star
    @u_pix_arr :: breath and width of the effective FoV
    @n_stars :: the number of neighbours the star will have
    @region_idx :: of the arr containing info about all regions with stars containing #n_star neighbours, we choose one region
    """
    # gets all the indicies of the stars in the FOV with n neighbours
    stars_with_n_neighbours = np.where(total_neighbours == n_stars)
    
    # check to make sure input is not outside range
    if region_idx >= len(stars_with_n_neighbours[0]):
        print('This file does not exist')
    else:
        fig, ax = plt.subplots(2,1, figsize=(11, 11), gridspec_kw={'height_ratios': [10, 5]})
        
        # plotting the entire data matrix
        flux_mat_full = np.load('Data/Many_star_model/flux_LSF_PSF_matrix2D.npy') 
        X, Y = np.linspace(0, u_pix_arr[0], u_pix_arr[0]), np.linspace(0, u_pix_arr[1], u_pix_arr[1])
        plot0 = ax[0].contourf(X, Y, flux_mat_full, cmap='YlGnBu')
        cbar = plt.colorbar(plot0, aspect=10, ax=ax[0]);
        
        # plot the star of concern
        ax[0].plot(x_FOV[stars_with_n_neighbours[0][region_idx]], y_FOV[stars_with_n_neighbours[0][region_idx]], 'k*', markersize=15)
        ax[1].plot(x_FOV[stars_with_n_neighbours[0][region_idx]], y_FOV[stars_with_n_neighbours[0][region_idx]], 'k*', markersize=15) 
        
        # plotting the cropped patch in the FoV of the data matrix
        limits = fap.limitsForCroppingMatrix(x_FOV, y_FOV, disperse_range, width, stars_with_n_neighbours[0][region_idx])        
        rect = Rectangle((limits[0], limits[2]), limits[1]-limits[0], limits[3]-limits[2],linewidth=1,edgecolor='r',\
                     facecolor='None', zorder=10, label='FOV')
        ax[0].add_patch(rect)
        setLabel(ax[0], '', 'y-axis position', 'Crop of a region with %d stars'%n_stars, 'default', 'default', legend=False)
        
        # zoom in plot of the cropped matrix
        save_data_crop_dir = 'Data/Cropped_Data_Images/'        
        flux_mat = np.load(save_data_crop_dir + '%d_stars/data_mat_%d.npy'%(n_stars, region_idx))        
        X_grid, Y_grid = np.meshgrid(np.linspace(limits[0], limits[1], len(flux_mat[0, :])), np.linspace(limits[2], limits[3], len(flux_mat[:, 0])))               
        plot1 = ax[1].contourf(X_grid, Y_grid, flux_mat, cmap='Blues')
        
        # labeling and setting colorbar
        setLabel(ax[1], 'x-axis position', 'y-axis position', '', 'default', 'default', legend=False)
        cbar = plt.colorbar(plot1, aspect=10, ax=ax[1]);
    return

def plotSubRegion(limits, flux_mat, title, x_label, cmaps):
    """
    Function to plot sub regions of the field of view
    @limits :: limits of the matrix in the defined FoV
    @flux_mat :: the 2D flux matrix that needs to be plotted
    @title, x_label :: axis properties for the plot
    @cmaps :: colors of the contour for the plot
    """
    fig, ax = plt.subplots(1,1, figsize=(11, 4))
    X_grid, Y_grid = np.meshgrid(np.linspace(limits[0], limits[1], len(flux_mat[0, :])), np.linspace(limits[2], limits[3], len(flux_mat[:, 0])))               

    plot1 = ax.contourf(X_grid, Y_grid, flux_mat, cmap=cmaps)

    # labeling and setting colorbar
    setLabel(ax, x_label, 'y-axis position (pixels)', title, 'default', 'default', legend=False)
    cbar = plt.colorbar(plot1, aspect=10);
    return 

def plotTargetStarPrediction(ax, hot_star_counts, target_star_prediction_all, hatch, chi_sq_cut, n_stars, plot_fig):
    """
    Function to plot a histogram showing the target star region for all stars for a given set of regions
    """
    # get the cold stars from the all star predictions array
    cold_star_counts, hot_probability, cold_probability = [], [], []
    
    for i in range(len(target_star_prediction_all)):
        cold_star_counts.append(len(target_star_prediction_all[i])-hot_star_counts[i])
        
        hot_probability.append((hot_star_counts[i]*100)/len(target_star_prediction_all[i]))
        cold_probability.append((cold_star_counts[i]*100)/len(target_star_prediction_all[i]))        
    
    if plot_fig:   
        labels = ['Cold stars Probability; chi-square cut = %d'%chi_sq_cut, 'Hot Star Probability; chi-square cut = %d'%chi_sq_cut]
        ax.hist(np.transpose([cold_probability, hot_probability]), stacked=True, label=labels, density=False, color=['#6b81d6', '#db7075'], hatch=hatch, edgecolor='k')
        
        setLabel(ax, 'Hot/Cold probability (%s) for %d stars'%('%', n_stars), 'Counts', 'Prediction for every target star', 'default', 'default', legend=True)
        ax.grid(False)
    return hot_probability, cold_probability

def plotChiSqPredictions(ax, cold_probability, hot_probability, linestyle, chi_sq_cut, n_stars):
    """
    Function to plot the chi-square predictions 

    """
    labels = ['Cold stars Probability; chi-square cut = %d'%chi_sq_cut, 'Hot Star Probability; chi-square cut = %d'%chi_sq_cut]
    ax.plot(cold_probability, '#db7075', linestyle=linestyle, label=labels[0])
    ax.plot(hot_probability, '#676a6e', linestyle=linestyle , label=labels[1])
    ax.hlines(50, 0, len(cold_probability), colors='k', linestyle='solid', linewidth=3, zorder=2)
            
    setLabel(ax, 'Target star number with %d neighbours'%(n_stars-1), \
        'Stellar type probability (%s)'%('%'), \
        'Predictions with chi-square cut = %d'%chi_sq_cut, 'default', [25, 75],\
         legend=True)
    return

def plotChiSqAllTemplates(ax, pal, resulting_params_all, hot_stars_arr, n_star):
    """
    Function to plot the chi-squares for all the templates

    """
    # choose the region 0
    star_idx = 0
    
    region_one_chi_sqs = np.concatenate(resulting_params_all[star_idx])
    norm_chi_sqs = region_one_chi_sqs/np.max(region_one_chi_sqs)
    print(r'There are %d number of total templates that were fitted to the region containing %d stars'%(len(norm_chi_sqs), n_star))
    
    num_perms = np.arange(len(norm_chi_sqs))
    len_pems = []
    
    for i, hot_stars in enumerate(hot_stars_arr):        
        len_pems.append(len(resulting_params_all[star_idx][i]))
        
        if i ==0: # for the first case
            ax.plot(num_perms[0:len_pems[-1]], norm_chi_sqs[0:len_pems[i]], marker='.', color=pal[i], label='$h_*$ = %d; $\ \Gamma (h_*)$ = %d'%(hot_stars*n_star, len(num_perms[0:len_pems[-1]])))
        
        # later cases, getting the start and end points and plotting them
        if i>0:  
            start = np.sum(len_pems[0:-1]).astype(int)
            x_axis = num_perms[start:start+len_pems[-1]]
            y_axis = norm_chi_sqs[start:start+len_pems[-1]]
            ax.plot(x_axis ,y_axis, '-', color=pal[i], label='$h_*$ = %d; $\ \Gamma (h_*)$ = %d'%(hot_stars*n_star, len(num_perms[0:len_pems[-1]])))
    ax.plot(num_perms, norm_chi_sqs, 'k.', markersize=2)
    
    # set labels
    setLabel(ax, 'Template numbers for fitting for $\mathcal{R}_0(n)$ ', 'Normalized best-fit estimator, $\mathbf{\Omega}$', '', [np.min(num_perms), np.max(num_perms)], 'default', legend=True)   
    ax.grid(False)
    plt.tight_layout()
    return

"""
Accuracy plots
""" 
def addColorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    Function taken from Matthias (https://bit.ly/39tuS5P)
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def plotColormap(fig, ax, colormap, acc_vals_arr, n_star_arr, num_lowest_temps):
    """
    Plot a colormap using imshow
    acc_vals_arr :: data array of shape (n, m) != 0
    n_star_arr, num_lowest_temps :: x and y axis labels arrays    
    """
    # set color map preferences and plot results
    cmap = plt.get_cmap(colormap)    
    psm = ax.imshow(np.transpose(acc_vals_arr), cmap=cmap)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(n_star_arr)))
    ax.set_yticks(np.arange(len(num_lowest_temps)))
    
    # and label them accordingly
    ax.set_xticklabels(n_star_arr)
    ax.set_yticklabels(num_lowest_temps)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(num_lowest_temps)):
        for j in range(len(n_star_arr)):
            text = ax.text(j, i, "%d"%(acc_vals_arr[j, i]*100), ha="center", va="center", color="k")
    
    # set the label for the plot
    setLabel(ax, r"Total number of stars in the region, $\mathcal{\mathbf{R}}_N$", r"Number of templates, $\beta$", 'Effect of statistics\n (number of templates chosen) on the accuracy of stellar type prediction\n', 'default', \
                'default', legend=False)
    
    addColorbar(psm)
    ax.grid(False)    
    return 

def plotAccuracyColormap(n_star_arr, num_lowest_temps, num_splits, colormap):
    """
    Function to collect the accuracy vals and plot it
    n_star_arr, num_lowest_temps :: x and y axis labels arrays
    num_splits :: sets the pixel scale (fitting parameter)
    colormap :: check matplotlib colormap for all possible options
    """
    # get these vals from files, set parameters
    acc_vals_arr = np.zeros((0, len(num_lowest_temps)))
    
    for n_s in n_star_arr:
        acc_vals = np.load('Data/Stellar_chi_sq_vals/%d_stars_accuracy_vals_%d_pixel_scale.npy'%(n_s, num_splits), allow_pickle=True)
        
        acc_vals_arr = np.append(acc_vals_arr, [acc_vals], axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plotColormap(fig, ax, colormap, acc_vals_arr, n_star_arr, num_lowest_temps)
    
    ax.set_title('Pixel Scale, $\Delta=$ %d'%num_splits)
    
    plt.savefig('Pictures/accuracy_map_for_diff_stats.png', facecolor='w')
    return acc_vals_arr

def plotAccuracyPixelScale(pixel_scale, num_lowest_temps, colormap, n_star_arr):
    """
    Function plots an accuracy color map for different pixel scales (a fitting parameter)
    n_star_arr, num_lowest_temps :: x and y axis labels arrays
    colormap :: check matplotlib colormap for all possible options    
    """
    fig, ax = plt.subplots(len(n_star_arr), 1, figsize=(8, 9*len(n_star_arr)))
    
    for i, n_s in enumerate(n_star_arr):
        # get these vals from files, set parameters
        acc_vals_arr = np.zeros((0, len(num_lowest_temps)))

        for p_s in pixel_scale:
            acc_vals = np.load('Data/Stellar_chi_sq_vals/%d_stars_accuracy_vals_%d_pixel_scale.npy'%(n_s, p_s), allow_pickle=True)
            acc_vals_arr = np.append(acc_vals_arr, [acc_vals], axis=0)

        if len(n_star_arr)>1:
            ax_id = ax[i]

        plotColormap(fig, ax_id, colormap, acc_vals_arr, pixel_scale, num_lowest_temps)
        ax_id.set_title('%d stars'%n_s)
        ax_id.set_xlabel('Pixel Scale, $\Delta$')
    return