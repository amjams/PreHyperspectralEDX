import numpy as np
from sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt
from scipy.optimize import nnls 
from matplotlib import cm
import cv2 as cv
import os
from scipy import signal
import math
import hyperspy.api as hs
import copy


def load_EDX(file_path, first_frame=0, last_frame = None, sum_frames=True, select_type=None, haadf_last_frame=True): 
    """wrapper for loading EMD data from hyperspy
    Parameters
    ----------
    see hyperspy's load
    
    Returns
    -------
    EDX: array, EDX dataset dimensions (h,w,b)
    HAADF: array dimension (h,w)
    xray_energies: array (b,)
    """
    s = hs.load(file_path,
                SI_dtype='uint8',
                first_frame=first_frame,
                last_frame=last_frame,
                sum_frames=sum_frames,
                select_type = select_type,
                load_SI_image_stack = True)    
    # search 
    #for i in range(len(s)):
        #if '2048, 2048' in repr(s[i]) and 'EDSTEMSpectrum' in repr(s[i]):   
            #EDX_idx = i
        #elif 'HAADF' in repr(s[i]): 
            #haadf_idx = i
    indices = [[i.metadata.General.title, len(i.axes_manager.shape)] for i in s]
    EDX_idx = indices.index(['EDS', 3])
    haadf_idx = indices.index(['HAADF', 3])

    # assign   
    if haadf_last_frame is True:
        haadf = s[haadf_idx].data[-1,:,:]
    elif haadf_last_frame is False:
        haadf = s[haadf_idx].data[:,:,:]
    elif isinstance(haadf_last_frame, int) is True:
        haadf = s[haadf_idx].data[haadf_last_frame,:,:]
    else:
        raise ValueError(f"Invalid haadf_last_frame value: {haadf_last_frame}")
             
    EDX = s[EDX_idx].data   
    xray_energies = s[EDX_idx].axes_manager.signal_axes[0].axis

    return EDX, haadf, xray_energies



def compute_inner_outer_similarity_with_distances(dist_to_ref, labels):
    """ Evaluation

    
    Returns
    -------
 
    """

    
    return similarity_dissimilarity_ratio


# Euclidean distance
def euc(array1, array2):
    return np.sqrt(np.sum((array1 - array2)**2))




    
    





def SAD(s1, s2):   
    """
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: `numpy array`
            The first vector.

        s2: `numpy array`
            The second vector.

    Returns: `float`
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math doesn't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle








        
def sparsity(spectrum):
    NumelSpectrum = spectrum.shape[0]*spectrum.shape[1]*spectrum.shape[1]
    return 100-(np.count_nonzero(spectrum)/NumelSpectrum)*100


