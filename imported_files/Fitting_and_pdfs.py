# -*- coding: utf-8 -*-
"""fitting_and_pdfs.py

##  Functions for fitting the templates to the data, generating data, and plotting the chi-squared distribution for different configurations of stellar population


**Author**: Soumya Shreeram <br>
**Supervisors**: Nadine Neumayer, Francisco Nogueras-Lara <br>
**Date**: 2nd November 2020

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
import Slitless_spec_forward_modelling as ssfm
import plotting as pt
import star_by_star_template_library as stl
"""

### 1. Functions for fitting the templates from the generated library

"""

def dataMatrix10StarsCase(template_dir, hot_stars, num_stars, u_pix, l_pix, x_pos, y_pos, disperse_range, dispersion_angle, templates_exist):
    """
    Function to build a data matrix from the template library for a given distribution of stars
    @template_dir :: directory where the templates are stored
    @hot_stars :: percent of hot stars in the FOV
    @num_stars :: number of stars in the FOV
    @l_pix, u_pix :: lower and upper limit in pixels defining the FOV
    @x_pos, y_pos :: the x and y position of the star  
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    @dispersion_angle :: variable sets the orientation of the dispersion
    @templates_exist :: boolean to check if the templates exist for the given population
    
    Returns ::
    @data_LSF_2Dmat :: 2darray containing the so-called data
    """
    # Load the 10 random spectra with added LSF, which can be associated with the stars in the FOV.
    flux_LSF2D, params = np.load('Data/flux_K2D_LSF.npy'), np.load('Data/params.npy')
    waves_k = np.load('Data/waves_k.npy')
    
    # division of stars chosen in FOV, considering 10 temperatures with 10 log g's
    stars_divide = ssfm.decideNumHotStars(hot_stars=hot_stars)
    
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    x_disperse, y_disperse = pt.plotDispersedStars(ax, x_pos, y_pos, disperse_range, waves_k, \
                                               dispersion_angle, no_plot=False)
    
    flux_k2D, type_id = ssfm.associateSpectraToStars(waves_k, stars_divide, num_stars, \
                                      flux_LSF2D, params)

    # generate all possible permutations that exist for the given distribution of stars
    if templates_exist:
        perms_arr = ssfm.constructSpectralTemplates10Stars(u_pix, y_disperse, x_disperse, \
                                                    type_id, flux_k2D)

    # building a data image that considers all possible (distinguishable) permuations
    perms_arr = np.load(template_dir+'hot_stars%d/perm_arr.npy'%(hot_stars*num_stars))
    data_LSF_2Dmat, perm_no = ssfm.constructDataImage10Stars(perms_arr, u_pix, flux_k2D, \
                                            y_disperse, x_disperse)
    return data_LSF_2Dmat, perm_no

def limitsForCroppingMatrix(x_FOV, y_FOV, disperse_range, width, idx):
    """
    Function to set the dimensions for cropping the flux matrix
    @x_FOV, y_FOV ::
    @disperse_range ::
    @width ::
    @idx ::
    """
    if x_FOV[idx]-disperse_range < 0:
        x_limit_start = 0
    else:
        x_limit_start = x_FOV[idx]-disperse_range
    
    limits = [x_limit_start, x_FOV[idx]+disperse_range, y_FOV[idx]-width/2, y_FOV[idx]+width/2]
    return limits


def cropMat(limits, X, Y, flux_mat):
    """
    Function to crop the matrix given the limits, X, Y pixel arrays and flux_mat (2D)
    """
    select_idx_x = np.where((X>limits[0]) & (X<limits[1]))
    select_idx_y = np.where((Y>limits[2]) & (Y<limits[3]))
    
    flux_mat = flux_mat[select_idx_y[0],:]
    flux_mat = flux_mat[:, select_idx_x[0]]
    return flux_mat

def cropDataMatrixNStarsCase(disperse_range, width, idx, x_FOV, y_FOV, u_pix_arr):
    """
    Function to build a data matrix from the template library for a given distribution of stars
    @disperse_range :: directory where the templates are stored
    @l_pix, u_pix :: lower and upper limit in pixels defining the FOV
    @x_pos, y_pos :: the x and y position of the star  
    @disperse_range :: range of wavelength in pixels chosen for dispersion
    @dispersion_angle :: variable sets the orientation of the dispersion
    @templates_exist :: boolean to check if the templates exist for the given population
    
    Returns ::
    @data_LSF_2Dmat :: 2darray containing the so-called data
    """
    # Load the data matrix
    flux_mat = np.load('Data/Many_star_model/flux_LSF_PSF_matrix2D.npy') 
    flux_PSF_mat = np.load('Data/Many_star_model/flux_PSF_matrix2D.npy')
    
    X, Y = np.linspace(0, u_pix_arr[0], u_pix_arr[0]), np.linspace(0, u_pix_arr[1], u_pix_arr[1])
    
    # define the limits of the region to be cropped
    limits = limitsForCroppingMatrix(x_FOV, y_FOV, disperse_range, width, idx)
        
    # selecting the y-band    
    select_idx_x = np.where((X>limits[0]) & (X<limits[1]))
    select_idx_y = np.where((Y>limits[2]) & (Y<limits[3]))

    flux_mat = cropMat(limits, X, Y, flux_mat)
    flux_PSF_mat = cropMat(limits, X, Y, flux_PSF_mat)
    return flux_mat, flux_PSF_mat, limits

def fitTemplateLibrary10Stars(template_dir, hot_stars_arr, num_stars, data_LSF_2Dmat, num_splits, u_pix):
    """
    Function to test all possible permutations for all configurations (stellar populations) of hot stars to the data image
    @template_dir :: directory where the templates are stored
    @hot_stars_arr :: arr holding info about the different configurations of hot stars considered
    @num_stars :: number of stars in the FOV
    @data_LSF_2Dmat :: 2darray containing the so-called data
    @num_splits :: reduced dimension of the flux matrix
    @u_pix :: upper limit in pixels defining the FOV
    
    """
    # store the chi-square value for every template
    chi_square3D = []
    # reduce the dimensions of the matrix to cal. the minimizing statistic 
    data_vals_2Darr = ssfm.splitFOVtoGrid(data_LSF_2Dmat, num_splits)
    
    for idx, stars in enumerate(hot_stars_arr):
        print('\n\nFor %d hot stars:\n'%(stars*num_stars))
        perms_arr = np.load(template_dir+'hot_stars%d/perm_arr.npy'%(stars*num_stars))
    
        # cal the chi-squared and the minimum chi-squared configuration
        chi_squared_arr = ssfm.determineBestFit10Stars(u_pix, num_stars, perms_arr, data_vals_2Darr, num_splits, many_stars=False)
        chi_square3D.append(chi_squared_arr)            
    return chi_square3D

def reduceMatrixResolutions(data_mat, limits, num_splits):
    """
    Function to reduce the resolution of the matrix for minimization
    @data_mat :: the 2D matrix
    @limits :: limits of the matrix in the defined FoV
    """
    data_mat_grid = np.zeros((0, int(data_mat.shape[1]/num_splits)))
    arr_x = np.linspace(limits[0], limits[1], int(data_mat.shape[1]))

    for i in range(data_mat.shape[0]):
        # split the 'splited flux matrix' further, but this time by columns
        binned_sums = ss.resamplingSpectra(arr_x, data_mat[i], int(data_mat.shape[1]/num_splits), 'sum')
        
        # store the sum of these vals 
        data_mat_grid = np.append(data_mat_grid, [binned_sums], axis=0)
    arr_x_new = np.linspace(limits[0], limits[1], int(data_mat.shape[1]/num_splits))
    return data_mat_grid, arr_x_new

def generateTemplatesCalChiSquare(x, y, mKs, template_dir, save_data_psf_crop_dir, n_stars, hot_stars, perms, disperse_range, dispersion_angle, u_pix_arr, limits, data_mat_arr, i, x_FOV, num_splits):
    """
    IMPORTANT: Function to generate and fit templates 
    """
    data_mat_grid, data_mat, data_PSF_mat = data_mat_arr[0], data_mat_arr[1], data_mat_arr[2]
    chi_squared = []

    for perm_idx in range(len(perms)): 
        diff_vals = 0
        flux_mat = ssfm.loadGenerateTemplates(template_dir, save_data_psf_crop_dir, n_stars, hot_stars,\
                                              [x, y, mKs], perm_idx, disperse_range, dispersion_angle, \
                                              u_pix_arr, limits)

        X_grid, Y_grid = np.linspace(0, u_pix_arr[0], u_pix_arr[0]), np.linspace(0, u_pix_arr[1], u_pix_arr[1])
        flux_mat_cropped = cropMat(limits, X_grid, Y_grid, flux_mat)
        flux_template = ss.addNoise(data_PSF_mat+flux_mat_cropped, [data_mat.shape[1], data_mat.shape[0]])
        
        # reduce the dimensions of the matrix to cal. the minimizing statistic
        flux_template_grid, arr_x = reduceMatrixResolutions(flux_template, limits, num_splits)
        
        
        # calculate statistic betweek 'data image and template image'
        for row in range(flux_template_grid.shape[0]):
            
            # calculate the chi-squared for the region to the right of the star only
            for col in np.where(arr_x > x_FOV)[0]:
                model = flux_template_grid[row][col]
                data = data_mat_grid[row][col]
                diff_vals += (model-data)**2
                
        # cal the final chi-sq statistic for each template and find the minimum
        chi_squared.append(np.sum(diff_vals))
    min_idx = np.where(chi_squared == np.min(chi_squared))[0][0]
    return chi_squared, np.min(chi_squared), min_idx

def loadDataReduceResolution(n_stars, i, num_splits, limits, save_data_crop_dir, save_data_psf_crop_dir):
    """
    """
    # load PSF matrix and data matrix for the region for the given number of neighbours
    data_mat = np.load(save_data_crop_dir + '%d_stars/data_mat_%d.npy'%(n_stars, i))
    data_PSF_mat = np.load(save_data_psf_crop_dir + '%d_stars/data_mat_PSF_%d.npy'%(n_stars, i))
        
    # reduce the dimensions of the matrix to cal. the minimizing statistic
    data_mat_grid, x_pixels = reduceMatrixResolutions(data_mat, limits[i], num_splits)
    return data_mat_grid, data_mat, data_PSF_mat 

def recoverBestTemplatePermutation(resulting_params_all, hot_stars_arr, i, n_stars, best_fit_perms_2D, template_dir, plot_chi_sqs):
    """
    DEPRECATED: Function to recover the best termplate permutation
    """
    resulting_params = resulting_params_all[i]

    # extracting the chi-square and best indicies
    chi_squares = [resulting_params[i][0] for i in range(len(hot_stars_arr))]
    all_template_nos = [resulting_params[i][1] for i in range(len(hot_stars_arr))]

    # normalizing chi-square, the min chi-sq index, and the best template number
    chi_squares_norm = chi_squares/np.max(chi_squares)
    best_idx = np.where(chi_squares == np.min(chi_squares))[0]    
    template_no = np.array(all_template_nos)[best_idx]

    # get the perm arr for the best hot-star distribution
    hot_stars = hot_stars_arr[best_idx]*n_stars
    
    best_fit_perm = np.load(template_dir+ '%d_stars/%d_hot_stars/perm_arr.npy'%(n_stars, hot_stars[0]))
    best_fit_perm = best_fit_perm[template_no.astype(int)]
    
    # save all the best perms for every region
    best_fit_perms_2D = np.append(best_fit_perms_2D, [np.array(best_fit_perm[0])], axis=0)

    if plot_chi_sqs:
        fig, ax = plt.subplots(1,1, figsize=(9, 8))
        ax.plot(hot_stars_arr,  chi_squares_norm, label='Region %d'%i)
        ax.plot(hot_stars_arr[best_idx],  np.min(chi_squares_norm), "r*") 
    ax = None
    return ax, best_fit_perms_2D

        
def analyzeAllChiSqs(ax, resulting_params_all, star_idx, hot_stars_arr, pal, plot_fig, method):
    """
    Function to evaluate chi-sqs and define the cut for analysis
    """
    region_one_chi_sqs = np.concatenate(resulting_params_all[star_idx])
    norm_chi_sqs = region_one_chi_sqs/np.max(region_one_chi_sqs)
    
    # define the chi-squre below which the templates are considered
    if method[0] == 'define_cut':
        chi_sq_cut = method[1]
        define_cut = (np.max(norm_chi_sqs)-np.min(norm_chi_sqs))/chi_sq_cut
        chi_sq_cut = np.min(norm_chi_sqs) + define_cut 
        selected_idicies = np.where(norm_chi_sqs < chi_sq_cut)
        
    if method[0] == 'find_N_smallest_vals':
        num_templates = method[1]
        selected_idicies = [np.argsort(norm_chi_sqs)[:int(num_templates)]]
        
    # plot only the first region's chi-square if plot_fig is set True
    if plot_fig and star_idx==0:
        pt.plotChiSqAllTemplates(ax, pal, norm_chi_sqs, hot_stars_arr, resulting_params_all, star_idx, chi_sq_cut)
    return norm_chi_sqs, selected_idicies

def findMinChiSq(resulting_params_all, star_idx):
    """
    Function to evaluate chi-sqs and define the cut for analysis
    """
    region_one_chi_sqs = np.concatenate(resulting_params_all[star_idx])
    norm_chi_sqs = region_one_chi_sqs/np.max(region_one_chi_sqs)
       
    # find the minimum chi-square
    min_chi_sq = np.min(norm_chi_sqs)
    return min_chi_sq, np.where(norm_chi_sqs == min_chi_sq)

def recoverBestTemplatePermutation(resulting_params_all, hot_stars_arr, i, n_stars, best_fit_perms_2D, template_dir, plot_chi_sqs):
    """
    Function 
    """
    resulting_params = resulting_params_all[i]    
    # extracting the chi-square and best indicies
    chi_squares = [resulting_params[i][0] for i in range(len(hot_stars_arr))]
    all_template_nos = [resulting_params[i][1] for i in range(len(hot_stars_arr))]    
    
    # normalizing chi-square, the min chi-sq index, and the best template number
    chi_squares_norm = chi_squares/np.max(chi_squares)
    best_idx = np.where(chi_squares == np.min(chi_squares))[0]    
    template_no = np.array(all_template_nos)[best_idx]    
    
    # get the perm arr for the best hot-star distribution
    hot_stars = hot_stars_arr[best_idx]*n_stars    
    best_fit_perm = np.load(template_dir+ '%d_stars/%d_hot_stars/perm_arr.npy'%(n_stars, hot_stars[0]))
    best_fit_perm = best_fit_perm[template_no.astype(int)]    
    
    # save all the best perms for every region
    best_fit_perms_2D = np.append(best_fit_perms_2D, [np.array(best_fit_perm[0])], axis=0)    
    if plot_chi_sqs:
        fig, ax = plt.subplots(1,1, figsize=(9, 8))
        ax.plot(hot_stars_arr,  chi_squares_norm, label='Region %d'%i)
        ax.plot(hot_stars_arr[best_idx],  np.min(chi_squares_norm), "r*") 
    ax = None
    return ax, best_fit_perms_2D

def evaluateAccuraryNstars(hot_probability, cold_probability, target_star_type):
    """
    Function to evaluate the accuracy of target stars
    @hot_probability :: percentage that defines the probability for the star to be hot 
    @cold_probability :: percentage that defines the probability for the star to be cold
    """
    stellar_type_predicts = []
    for i in range(len(hot_probability)):
        if hot_probability[i] > cold_probability[i]:
            stellar_type_predicts.append(0)
        elif hot_probability[i] == cold_probability[i]:
            stellar_type_predicts.append(None)
        else:
            stellar_type_predicts.append(7)

    count_good_stars = 0
    for i in range(len(stellar_type_predicts)):
        if stellar_type_predicts[i] == target_star_type[i]:
            count_good_stars += 1

    return count_good_stars/len(stellar_type_predicts), np.where(stellar_type_predicts == target_star_type),  np.where(stellar_type_predicts != target_star_type)

def checkAccuracyForMinChiSqTechnique(min_target_star_prediction_all, target_star_type):
    """
    Function to check the accuracy of the minimum chi-sq method
    """
    count_good_stars = 0
    for i in range(len(min_target_star_prediction_all)):
        if min_target_star_prediction_all[i] == target_star_type[i]:
            count_good_stars += 1
    return count_good_stars

def extractingAccuracyRoutine(n_stars, total_neighbours, hot_stars_arr, template_dir):

    # associate spectra to those stars
    stars_with_n_neighbours = np.where(total_neighbours == n_stars)
    
    # load files
    resulting_params_all = np.load('Data/Chi_sq_vals/%d_stars_%d_regions.npy'%(n_stars, len(stars_with_n_neighbours[0])), allow_pickle=True)

    stellar_types_data_arr = np.load('Data/Target_star_predictions/Data_stellar_info_%d_stars.npy'%n_stars, allow_pickle = True)
    target_star_data =  np.load('Data/Target_star_predictions/Data_target_info_%d_stars.npy'%n_stars, allow_pickle = True)
    target_star_type, target_star_idx = target_star_data[0], target_star_data[1]
    
    for j, chi_sq_cut in enumerate([4]):
        hot_star_counts, target_star_prediction_all = stl.getTargetStarPrediction('ax', hot_stars_arr, resulting_params_all, stars_with_n_neighbours,\
         template_dir, n_stars, target_star_idx, print_msg=False, chi_sq_cut=chi_sq_cut)

        hot_probability, cold_probability = pt.plotTargetStarPrediction('ax', hot_star_counts, target_star_prediction_all, 'hatch[j]', chi_sq_cut, n_stars, plot_fig=False)

        accuracy = evaluateAccuraryNstars(hot_probability, cold_probability, target_star_type)
    return accuracy