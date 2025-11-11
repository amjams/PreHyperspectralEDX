import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2 as cv
import os
from scipy import signal
import math
import hyperspy.api as hs
import copy
import tifffile as tif




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


    
def mean_filter(img, kernel_size=3):
    """ Apply a mean filter to an image. 

        Parameters
        ----------
        kernel_size: size (width/height) of the kernel
    """
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)
    return cv.filter2D(img,-1,kernel) 



def MinMax(data):
    # minmax the data (0-1 normalize)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def Normalize_uint8(img, normalize_by=None):
    """Normalize an image then set to uint8.

    Parameters
    ----------
    normalize_by : optional
        Another array to normalize by.
    """
    if normalize_by is None:
        mn = img.min()
        mx = img.max()
    else:
        mn = normalize_by.min()
        mx = normalize_by.max()

    img_out = ((img-mn) / (mx-mn)) * 255
    return img_out.astype(np.uint8)


def SAD(s1, s2, eps=1e-12):
    """
    Computes the spectral angle mapper (SAD) between two spectra or arrays of spectra.

    Parameters
    ----------
    s1 : np.ndarray
        First spectrum or array (..., bands)
    s2 : np.ndarray
        Second spectrum or array (..., bands)
    eps : float, optional
        Small constant to avoid division by zero (default 1e-12)

    Returns
    -------
    angle : np.ndarray or float
        Spectral angle(s) in radians.
    """
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    dot = np.sum(s1 * s2, axis=-1)
    norm1 = np.linalg.norm(s1, axis=-1)
    norm2 = np.linalg.norm(s2, axis=-1)

    cos_theta = dot / (norm1 * norm2 + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)

def binning_xy(img, dim = None):
    """
    General purpose binning for images in xy.

    Parameters
    ----------
    img : 2D numpy array
    dim : dimensions after binning

    Returns
    -------
    angle : np.ndarray or float
        Spectral angle(s) in radians.
    """
    
    # old and new dimensions
    old_y, old_x = img.shape
    
    if dim is None:
        dim = [int(old_y/2),int(old_x/2)]
    
    if isinstance(dim, list) is False:
        dim = [dim, dim]
    
    if any(img.shape[i] % dim[i] != 0 for i in range(2)):
        raise ValueError("Ensure old dims are divisible by new dims.")
    
    new_y, new_x = dim
    
    # binning factors
    fy, fx = old_y // new_y, old_x // new_x
 
    
    return img.reshape(new_y, fy, new_x, fx)


def create_masks(mask_dir):
    """
    get the mask image array from a directory containing 
    tif files (one binary image per annotated class)

    Parameters
    ----------
    mask_dir: path to the folder containing the tifs

    Returns
    -------
    masks: 2D numpy array containing the annotations 
    (0: background, 1,..,n: annotations)
    """

    file_names = os.listdir(mask_dir)
    file_names = [name for name in file_names if name.endswith('tif')]
    file_names.sort()
    mask_paths = [os.path.join(mask_dir, file_name) for file_name in file_names]

    # get dimensions
    d1, d2 = tif.imread(mask_paths[0]).astype('bool').shape

    masks = np.zeros((d1, d2), dtype='bool')

    for i in range(len(mask_paths)):
        x = tif.imread(mask_paths[i]).astype('bool')
        masks = masks + x * (i + 1)

    return masks


def sil_scores(hsi, masks, metric='euclidean'):
    """
    Compute the sillhouette scores of clusters associated
    with annotated masks over a hyperspectral image (hsi)
    
    Parameters
    ----------
    hsi : 3D numpy array
    masks : 2D numpy array, same 2D dims as hsi
        clusters annotated as 1,..,n and background as 0
    metric : metric for the silhouette score
    
    Returns
    -------
    sil_img : 2D numpy array of silhouette scores 
              (NaN for non-annotated pixels)
    """

    # get hsi dimensions
    h, w, b = hsi.shape

    # hsi flattened
    hsi_flat = hsi.reshape((h*w,b))
    
    # sillouhette value per sample
    masks_flat = masks.reshape((h*w,)).astype(int)
    annotated_idx = np.where(masks_flat!=0)[0]
    labels = masks_flat[annotated_idx]
    sil_values = silhouette_samples(hsi_flat[annotated_idx,:],labels, metric=metric)

    # create a sillhouette img
    sil_img = np.full((h * w,), np.nan, dtype=float)

    # put silhouette values back into their original positions
    sil_img[annotated_idx] = sil_values

    return sil_img.reshape((h, w))


    



    



