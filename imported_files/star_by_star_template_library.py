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
    x_start = start_point[0]
    
    for idx in range(len(mag_Ks)):
        # limits to define the neighbours of a star
        limits = [x_start, x_pos[idx]+disperse_range, y_pos[idx]-width/2, y_pos[idx]+width/2]
        
        # within FOV
        x, y, mKs, mH = ssfm.selectFOV(limits, x_pos, y_pos, mag_H, mag_Ks, print_msg=False)
                
        # influencing stars outside FOV
        x_pos_full, y_pos_full, mag_Ks_full, mag_H_full = selected_c_pxls[0], selected_c_pxls[1], selected_c_pxls[2], selected_c_pxls[3]
        
        # the inside+outside stars that are influencing from theFOV
        x_out, y_out, mKs_out, mH_out  = ssfm.selectFOV(limits, x_pos_full, y_pos_full, mag_Ks_full, mag_H_full, print_msg=False)
        
        # save positions and mag info of the stars in and out of the FOV
        star_neighbours_in.append([x, y, mKs, mH])        
        star_neighbours_out.append([x_out, y_out, mKs_out, mH_out])
        
        # save stars in and out of the FOV
        num_neighbours_in.append(len(mKs))
        num_neighbours_out.append(len(mKs_out))
        
    return [star_neighbours_in, star_neighbours_out], [num_neighbours_in, num_neighbours_out]

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

def starsOutsideFOV(star_neighbours, x_pos, y_pos, mag_H, mag_Ks, foreground_cutoff, start_points, u_pix, selected_c_pxls):
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

def defineEffectiveFoV(stars_outside_FOV, x_start, u_pix):
    """
    Function defines the edges of the FOV
    """
    left_edge = x_start - np.min(stars_outside_FOV[2])
    right_edge = np.max(stars_outside_FOV[2])-x_start-u_pix
    
    effective_width = left_edge + u_pix + right_edge
    return int(effective_width), u_pix

def dispersionWithNstars(total_stars_FOV_params, u_pix_arr, waves_k, disperse_range, dispersion_angle, create_dispersed_files):
    """
    Function to disperse the stars in the effective FOV
    Note that the effective FOV also contains stars close to the edges, whose info can't be obtained accurately 
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



def addPSFtoStarsOutsideFOV():
    """
    Function to add PSF to all stars outside the FOV
    """
    return 
