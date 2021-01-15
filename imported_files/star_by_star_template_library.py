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

# for doing combinatorics with spectra
from itertools import permutations
from more_itertools import distinct_permutations

# personal imports
import Simulating_Spectra as ss
import Slitless_spec_forward_modelling as ssfm
import plotting as pt
import Fitting_and_pdfs as fap

"""

### 1. Functions for reducing to a sub-grid

"""

def findStarAndNeighbours(x_pos, y_pos, mag_H, mag_Ks, disperse_range, width, selected_c_pxls, start_point, u_pix):
    """
    Function to find the star and it's nearest neighbours in what is defined as the 'Field of Influence' (FoI)
    @x_pos, y_pos :: arr of stars
    """
    star_neighbours_in, num_neighbours_in = [], []
    star_neighbours_out, num_neighbours_out = [], []
    x_start, y_start = start_point[0], start_point[1]
    
    limits_FOV = [x_start, x_start+u_pix, y_start, y_start+u_pix]
    x_FOV, y_FOV, mH_FOV, mKs_FOV = ssfm.selectFOV(limits_FOV, x_pos, y_pos, mag_H, mag_Ks, print_msg=False)

    for idx in range(len(mH_FOV)):
        # limits to define the neighbours of a star
        limits_neighbours = [x_FOV[idx]-disperse_range, x_FOV[idx]+disperse_range, y_FOV[idx]-width/2, y_FOV[idx]+width/2]
        
        # FOV
        x, y, mKs, mH = ssfm.selectFOV(limits_neighbours, x_FOV, y_FOV, mH_FOV, mKs_FOV, print_msg=False)     
               
        # the inside+outside stars that are influencing from theFOV
        x_out, y_out, mKs_out, mH_out  = ssfm.selectFOV(limits_neighbours, x_pos, y_pos, mag_H, mag_Ks, print_msg=False)
        
        # save positions and mag info of the stars in and out of the FOV
        star_neighbours_in.append([x, y, mKs, mH])        
        star_neighbours_out.append([x_out, y_out, mKs_out, mH_out])
        
        # save stars in and out of the FOV
        num_neighbours_in.append(len(mKs))
        num_neighbours_out.append(len(mKs_out))
        
    return [star_neighbours_in, star_neighbours_out], [num_neighbours_in, num_neighbours_out], x_FOV, y_FOV

def recordStellarTypesOfARegion(x_in, y_in, x_pos, y_pos, type_id, x_FOV, y_FOV, stars_idx):
    """
    Function to record the stellar types of stars within the defined FoV
    """
    select_idx = []
    # select desired region
    for i in range(len(x_in)):
        counts = np.where((x_pos == x_in[i]) & (y_pos == y_in[i]))
        select_idx.append(counts[0][0])
       
    target_star = np.where((x_pos == x_FOV[stars_idx]) & (y_pos == y_FOV[stars_idx]))
    target_star_idx = np.where((x_in == x_FOV[stars_idx]) & (y_in == y_FOV[stars_idx]))
    return type_id[select_idx], type_id[target_star], target_star_idx[0]

def generateDirToSavePerms(template_dir, num_stars, hot_stars):
    """
    Function automatically makes the directory to save all the files
    """
    folder0 = os.path.join(template_dir,'%d_stars'%num_stars)
    folder1 = os.path.join(folder0,'%d_hot_stars'%hot_stars)
    
    if not os.path.exists(folder0):
        os.mkdir(folder0)
        os.mkdir(folder1)
    
    if os.path.exists(folder0) and not os.path.exists(folder1):            
            os.mkdir(folder1)       
    return

def generateDirToSaveDataCrops(save_dir, num_stars):
    """
    Function automatically makes the directory to save all the files
    """
    folder0 = os.path.join(save_dir,'%d_stars'%num_stars)
    
    if not os.path.exists(folder0):
        os.mkdir(folder0)       
    return

def generateSaveTemplates(type_id, flux_LSF2D, flux_k2D, template_dir, num_stars, hot_stars):
    """
    Function to generate and save the templates 
    @perms :: 2d array holding all permutation possible for a given ensemble
    @flux_LSF2D :: arr holding 10 different spectra (from hot to cold stars)
    @flux_k2D :: arr with spectra = number of stars with 2 types of stars (hot and cold)
    @hot_stars :: percent of hot stars among 
    @num_stars :: total number of neighbouring stars
    @template_dir :: directory where the templates are stored
    """
    # generate all possible permutations that exist for the given distribution of stars
    perms = distinct_permutations(type_id)
    perms = list(perms)
        
    # generate the directory to save the permutations
    generateDirToSavePerms(template_dir, num_stars, hot_stars*num_stars)       

    for i in range(len(perms)):
        # reorder the flux arr for the given permutation
        flux_k2D = flux_LSF2D[np.array(perms[i])]

        # save the information that will be later useful to generate 2D templates
        np.save(template_dir+'%d_stars/%d_hot_stars/flux_%dperm.npy'%(num_stars, hot_stars*num_stars, i), flux_k2D)          
    
    # save the permutation array
    np.save(template_dir+'%d_stars/%d_hot_stars/perm_arr.npy'%(num_stars, hot_stars*num_stars), list(perms))    
    return

def starsOutsideFOV(star_neighbours):
    """
    Function to associate spectra to stars outside the selected FOV that yet influences the concerned stars
    @hot_stars_out :: variable to set the hot star popultation outside the FOV
    @foreground_cutoff :: cut off limit for distinguishing between foreground and gc stars
    """
    x_pos_out, y_pos_out, mag_H_out, mag_Ks_out = [], [], [], []
    for i in range(len(star_neighbours[1][:])):
        x_pos_out.append(star_neighbours[1][i][0])
        y_pos_out.append(star_neighbours[1][i][1])        
        mag_H_out.append(star_neighbours[1][i][2])
        mag_Ks_out.append(star_neighbours[1][i][3])
    
    # concatenate all the sub arrays
    x_pos_out_added, y_pos_out_added = np.concatenate(x_pos_out), np.concatenate(y_pos_out)
    
    print('Around %d stars form neighbours, of which most are repeats. In total, %d stars need to be modelled.'%( len(np.unique(x_pos_out_added)), len(x_pos_out_added) ))    
    
    return [x_pos_out, y_pos_out, x_pos_out_added, y_pos_out_added, mag_H_out, mag_Ks_out]


def dispersionWithNstars(total_stars_FOV_params, u_pix_arr, waves_k, disperse_range, dispersion_angle, create_dispersed_files):
    """
    Function to disperse all the stars in the effective FOV
    @total_stars_FOV_params :: arr that contains [x_pos_out, y_pos_out, x_pos_out_added, y_pos_out_added, mag_H_out, mag_Ks_out] for the effective FoV
    @u_pix_arr :: arr that contains the x-y dimension of the effective FoV
    @waves_k :: wavelength arr over which the stars are dispersed
    @disperse_range :: the length dispersion range for each star
    @dispersion_angle :: the angle of dispersion
    @create_dispersed_files :: bool to avoid regenerating the dispersion, if it already exists

    Returns ::
    x_disperse, y_disperse :: the indicies of dispersion of every star in the 'effective FoV'
    """    
    # mapping right ascension and declination to (pixel, pixel) FOV
    x_posFOV, y_posFOV = ssfm.mapToFOVinPixels(total_stars_FOV_params[2], total_stars_FOV_params[3], u_pix_arr[0], u_pix_arr[1])

    # dispersing the stars and plotting it
    fig, ax = plt.subplots(1,1,figsize=(16,8))

    if create_dispersed_files:
        x_disperse, y_disperse = pt.plotDispersedStars(ax, x_posFOV, y_posFOV, \
                                                       disperse_range, waves_k, \
                                                       dispersion_angle)
        # save the files that have been created
        np.save('Data/Many_star_model/x_disperse_FOV_%dx%d.npy'%(x_start, y_start), x_disperse)
        np.save('Data/Many_star_model/y_disperse_FOV_%dx%d.npy'%(x_start, y_start), y_disperse)
    else:
        x_disperse = np.load('Data/Many_star_model/x_disperse_FOV_%dx%d.npy'%(x_start, y_start))
        y_disperse = np.load('Data/Many_star_model/y_disperse_FOV_%dx%d.npy'%(x_start, y_start))

        # plot the dispersion
        for i in range(len(x_disperse)):
            ax.plot(x_disperse[idx], y_disperse[idx], '.', color='#d8b6fa')

        # and the stars that produce the dispersion
        ax.plot(total_stars_FOV_params[2], total_stars_FOV_params[3], ".", color= '#ebdf09', alpha=0.9, marker="*", markersize=10)

        # set the label for the plot
        pt.setLabel(ax, 'x-axis position', 'y-axis position', '', 'default', \
                'default', legend=False)
    return x_disperse, y_disperse


def findStellarTypesDataStars(n_stars, hot_stars_arr, stars_with_n_neighbours, star_neighbours, x_pos, y_pos, x_FOV, y_FOV):
    """
    Function to find the stellar types of the region from the DATA image
    @n_stars :: the number of stars withing the region of the data image
    @hot_stars_arr :: arr holds info about the % of hot stars considered
    @stars_with_n_neighbours :: arr that contains the indicies of all the target stars in the data image containing n_stars for neighbours
    @star_neighbours :: ndarray that holds info about [x, y, mKs, mH] for all the stars in the 'effective FoV'
    @x_pos, y_pos :: the x and y position of all the stars in the effective FoV
    @x_FOV, y_FOV :: as the name suggests, the x and y positions of all the stars in the 'defined FoV'

    Returns ::
    @stellar_types_data_arr :: this arr holds info about the configuration i.e. stellar types of all the stars in the region
    @target_star_type :: arr hold info about the stellar type of the target star along
    @target_star_idx :: arr holds info about the position of the target star index amongst the arr of all n_stars in the defined region
    """
    type_id = np.load('Data/Many_star_model/type_id_shuffled.npy')
    stellar_types_data_arr = np.zeros((0, n_stars)) 
    target_star_type = []
    target_star_idx = []
    for i, stars_idx in enumerate(stars_with_n_neighbours[0]):
        x, y, mKs = star_neighbours[1][stars_idx][0], star_neighbours[1][stars_idx][1], star_neighbours[1][stars_idx][3]

        # index 0 holds info about all the stellar types in the FoV, index 1 hold info about the target star type
        stellar_types_data = recordStellarTypesOfARegion(x, y, x_pos, y_pos, type_id, x_FOV, y_FOV, stars_idx)

        # at each iteration saves info about the stellar types of the region 
        stellar_types_data_arr = np.append(stellar_types_data_arr, [stellar_types_data[0]], axis=0)
        target_star_type.append(stellar_types_data[1][0])
        target_star_idx.append(stellar_types_data[2][0])

    target_star_info = [target_star_type, target_star_idx]
    np.save('Data/Target_star_predictions/Data_stellar_info_%d_stars.npy'%(n_stars), stellar_types_data_arr)
    np.save('Data/Target_star_predictions/Data_target_info_%d_stars'%n_stars, [target_star_type, target_star_idx])
    return stellar_types_data_arr, target_star_type, target_star_idx


def countHotStars(target_star_prediction):
    count_hot_stars = 0
    for m in range(len(target_star_prediction)):
        if target_star_prediction[m] == 0:
            count_hot_stars += 1
    return count_hot_stars


def findTemplateNumber(best_fit_perm_all, template_dir, selected_temps, resulting_params_all, hot_stars_arr, n_stars, region_idx):
    """
    Function to find the template no of the chosen configuration (templates are ordered in directories labelled by their hot star distribution)
    @template_dir :: the directory containing the template files
    @selected_temps :: arr containing the values of all the selected templated files
    @resulting_params_all :: arr containing the info of the chi-squares for all the templates, for each hot-star distribution under concern
    @hot_stars_arr :: arr holds info about the % of hot stars considered
    @n_stars :: the number of stars withing the region of the data image
    @region_idx :: index runs over all the regions that contain #n_stars

    Returns ::
    best_fit_perm_all :: arr holds info about all the configurations of the chosen templates
    """
    template_nos, len_pems, normalize_temps = [], [], []
    
    for i, hot_stars in enumerate(hot_stars_arr):
        normalize_temps = []
        len_pems.append(len(resulting_params_all[region_idx][i]))
        
        # for the first case            
        if i == 0: 
            if np.any(selected_temps[0] < len_pems[i]):
                # get the template nos
                chosen_vals = np.where(selected_temps[0] < len_pems[-1])
                normalize_temps = np.array(selected_temps[0])[chosen_vals]
                
        # for hot star distributioins that are greated than 0/Null
        if i>0:
            if np.any((selected_temps[0] >= np.sum(len_pems[0:-1])) & (selected_temps[0]< np.sum(len_pems))):
                # for all other hot-star distributions
                chosen_vals = np.where((selected_temps[0] >= np.sum(len_pems[0:-1])) & (selected_temps[0]< np.sum(len_pems)))
                normalize_temps = np.array(selected_temps[0])[chosen_vals]-np.sum(len_pems[0:-1])
        
        if len(normalize_temps)>0:
            # get the perm arr for the best hot-star distribution
            best_fit_perm = np.load(template_dir+ '%d_stars/%d_hot_stars/perm_arr.npy'%(n_stars, hot_stars*n_stars))
            best_fit_perm = best_fit_perm[normalize_temps.astype(int)]
            
            if best_fit_perm.shape[1] != n_stars:
                best_fit_perm = best_fit_perm[:, 0:n_stars]
            best_fit_perm_all = np.append(best_fit_perm_all, best_fit_perm, axis=0)
    return best_fit_perm_all

def getTargetStarPrediction(hot_stars_arr, resulting_params_all, stars_with_n_neighbours,\
 template_dir, n_stars, target_star_idx, print_msg, method):
    """
    Function to get the stellar type of all the target star
    @hot_stars_arr :: arr holds info about the % of hot stars considered
    @resulting_params_all :: arr containing the info of the chi-squares for all the templates, for each hot-star distribution under concern
    @stars_with_n_neighbours :: arr that contains the indicies of all the target stars in the data image containing n_stars for neighbours
    @template_dir :: the directory containing the template files
    @n_stars :: the number of stars withing the region of the data image
    @print_msg :: boolean to decide wether to print the message or not
    """
    target_star_prediction_all = []
    hot_star_counts = []
    best_fit_perm_all = np.zeros((0, n_stars))
    
    for i, stars_idx in enumerate(stars_with_n_neighbours[0]):    
        norm_chi_sqs, selected_temps = fap.analyzeAllChiSqs('ax', resulting_params_all, i, hot_stars_arr, pal=0, plot_fig=False, method=method)
        
        # find the stellar configurations of the best templates that are chosen
        best_fit_perm_all = findTemplateNumber(best_fit_perm_all, template_dir, selected_temps, resulting_params_all, hot_stars_arr, n_stars, region_idx=i)
        
        target_star_prediction = []
        for idx in range(len(best_fit_perm_all)):
            target_star_prediction.append(best_fit_perm_all[idx][target_star_idx[i]])
        
        # count the hot stars in the predictions array
        hot_star_counts.append(countHotStars(target_star_prediction))

        # save all the predictions
        target_star_prediction_all.append(target_star_prediction)
        
        if print_msg:
            print('Regions %d, %.2f hot star'%(i, hot_star_counts[i]/len(target_star_prediction)))

    # needed to find the total number of predictions made for a given star
    np.save('Data/Target_star_predictions/All_target_star_predictions_%d_star_regions.npy'%n_stars, target_star_prediction_all)

    # needed for accessing if a given star is hot or cold
    np.save('Data/Target_star_predictions/Hot_stellar_type_probability_%d_star_regions.npy'%n_stars, hot_star_counts)
    return hot_star_counts, target_star_prediction_all


def getMinimumChiSqStarPrediction(resulting_params_all, template_dir, hot_stars_arr, n_stars, stars_with_n_neighbours, target_star_idx):
    target_star_prediction_all = []
    hot_star_counts = []
    best_fit_perm = np.zeros((0, n_stars))
    
    for i, stars_idx in enumerate(stars_with_n_neighbours[0]):
        min_chi_sq, selected_temp_no = fap.findMinChiSq(resulting_params_all, i)

        # find the stellar configurations of the best templates that are chosen
        best_fit_perm = findTemplateNumber(best_fit_perm, template_dir, selected_temp_no, resulting_params_all, hot_stars_arr, n_stars, region_idx=i)
        
        # get the target star prediction and save it for every region
        target_star_prediction = best_fit_perm[0][target_star_idx[i]]
        target_star_prediction_all.append(target_star_prediction)
    return target_star_prediction_all