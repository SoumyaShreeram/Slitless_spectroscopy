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

# personal imports
import Simulating_Spectra as ss

"""### 1. Functions for reading in the catalog
"""

def readCatalogFile(filename):
    """
    Function to read the magnitude and positions of the stars from the catalog file
    """
    hdu_list = fits.open(filename)
    # Ks band magnitude
    mag_Ks = hdu_list[1].data['Ksmag']

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
    return mag_Ks, ra_Ks, de_Ks, errors


"""### 2. Functions for selection of stars
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
    Function cuts off stars below a certain flux from the bottom@de_Ks, ra_Ks, mag_Ks :: declination , right ascension, and magnitude of stars arr
    """
    
    idx = np.where(mag_Ks>cut_off_ll)
    print("Discarding stars with magnitude < %d."%cut_off_ll)
    return mag_Ks[mag_Ks>cut_off_ll], ra_Ks[idx], de_Ks[idx]

def selectMaxStars(mag_Ks, ra_Ks, de_Ks, max_stars):
    """
    Function selects #max_stars randomly within the FOV
    @de_Ks, ra_Ks, mag_Ks :: declination , right ascension, and magnitude of stars arr
    """
    
    idx = ss.generateRandInt(0, len(mag_Ks), max_stars)
    print("Selecting a max of %d stars in the FOV randomly."%max_stars)
    return mag_Ks[idx], ra_Ks[idx], de_Ks[idx]


