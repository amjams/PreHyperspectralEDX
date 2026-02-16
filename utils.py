import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
from pprint import PrettyPrinter


def testfun():
    ssim(np.zeros((50,50)),np.ones((50,50)),data_range=1)


def load_EDX(file_path, first_frame=0, last_frame = None, sum_frames=True, select_type=None, haadf_last_frame=True, return_dict=False, verbose = False): 
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
    if sum_frames:
        EDX_idx = indices.index(['EDS', 3])
    else:
        EDX_idx = indices.index(['EDS', 4])
        
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


    # Show what's inside
    if verbose:
        pp = PrettyPrinter(width=200,depth=None,compact=False)
        pp.pprint(s)

    if return_dict:
        EDX = s[EDX_idx]
    
    return EDX, haadf, xray_energies


    
def mean_filter(img, kernel_size=3):
    """ Apply a mean filter to an image. 

        Parameters
        ----------
        kernel_size: size (width/height) of the kernel
    """
    kernel = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)
    return cv.filter2D(img,-1,kernel) 



def MinMax(data, return_extra=False):
    # minmax the data (0-1 normalize)
    min = np.min(data)
    max = np.max(data)
    if return_extra:
        return (data - np.min(data)) / (np.max(data) - np.min(data)), min, max
    else: 
        return (data - np.min(data)) / (np.max(data) - np.min(data))


def OneMinusOne(data, return_extra=False):
    # -1 and 1 normalize
    minmax, min, max = MinMax(data, return_extra=True)
    
    if return_extra:
        return 2*minmax-1, min, max
    else: 
        return 2*minmax-1


def MinMaxInverse(MinMax, min, max):
    # return minmaxed data to original range
    return MinMax*(max-min)+min


def Normalize_uint8(img, normalize_by=None):
    """Normalize an image then set to uint8.

    Parameters
    ----------
    normalize_by : optional
        Another array to normalize by.
    """
    if normalize_by is None:
        mn = np.nanmin(img)
        mx = np.nanmax(img)
    else:
        mn = np.nanmin(normalize_by)
        mx = np.nanmax(normalize_by)

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
    binned image 
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
 
    
    return img.reshape(new_y, fy, new_x, fx).mean(axis=(1, 3))

def binning_xyz(EDX, dim = [2048,2048,250]):
        """
        General purpose binning for images in xyx.
    
        Parameters
        ----------
        img : 3D numpy array (EDX HSI)
        dim : dimensions after binning
    
        Returns
        -------
        binned HSI 
        """
        
        # original and new dimensions
        old_y, old_x, old_b = EDX.shape
        new_y, new_x, new_b = dim
    
        # binning factors
        fy, fx, fb = old_y // new_y, old_x // new_x, old_b // new_b
    
        # EDX 
        out = EDX.reshape(new_y, fy, new_x, fx, new_b, fb)
        out = out.mean(axis=(1, 3, 5))

        return out



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


def get_spectra(hsi, roi_pixel, size=3, all_spectra=False):
    """
    Return the spectra in (sizexsize) square centered around roi_pixel

    Parameters
    ----------
    hsi : 3D numpy array
    roi_pixel : (x,y) coordinates of the pixel, tuple or list.
    size : (size x size) of the square from which to return spectra.
    all_spectra : return all the spectra in the roi (default returns only the mean).

    Returns
    -------

    """

    return


def hsi_multi_scale(hsi,radii=[1],sigma=1):
    """
    Return the Gaussian extended (multi-scale) versions of the HSI

    See: https://github.com/amjams/HyperspectralEDX/blob/main/functions_EDX.py

    Parameters
    ----------
    hsi : 3D numpy array
    radii : Radii of the Gaussian filters
    sigma : Std. deviation of the Gaussian filter
    
    Returns
    -------

    """
    # get hsi dimensions
    h, w, b = hsi.shape
    
    # determine the largest radius
    maxr = np.max(np.asarray(radii))
    
    # initialize an array to the save the new features
    hsi_extended = np.zeros((h,w,b*(len(radii)+1)))

    hsi_extended[:,:,:b] = hsi
    
    for idx,r in enumerate(radii):
        idx = idx+1
        hsi_extended[:,:,b*idx:(b*idx)+b] = GaussFilterCube(hsi,size=r*2+1,sigma=2)
    
    return hsi_extended
    

def GaussFilter(im,apply=True,sigma = 2, size=3):
    """
    See: See: https://github.com/amjams/HyperspectralEDX/blob/main/functions_EDX.py
    """
    if apply:
        kernel = np.ones((size,size),np.float32)/(size*size)
        im_filtered = cv.GaussianBlur(im,(size,size),sigmaX = sigma, sigmaY= sigma, borderType =cv.BORDER_DEFAULT)
    else:
        im_filtered= im
    return im_filtered


def GaussFilterCube(hsi, sigma = 2, size=3):
    """
    See: See: https://github.com/amjams/HyperspectralEDX/blob/main/functions_EDX.py
    """
    hsi_filtered = np.zeros(hsi.shape)
    for i in range(hsi.shape[2]): 
        hsi_filtered[:,:,i] = GaussFilter(hsi[:,:,i],apply=True,sigma=sigma,size=size)
    return hsi_filtered


def eval_psnr(hsi_1, hsi_2):
    """
    Get the channel-wise psnr between two hyperspectral cube

    Returns:
    --------
    list of psnr values
    """

    # Get dimensions
    h,w,b = hsi_1.shape

    # Initialize list for scores
    psnr_list = []

    for k in range(b):
        img1 = MinMax(hsi_1[:,:,k])
        img2 = MinMax(hsi_2[:,:,k])
        psnr_list.append(psnr(img1,img2,data_range=1.0))

    return psnr_list
        
        

def eval_ssim3(hsi_1, hsi_2):
    """
    Get the channel-wise SSIM between two hyperspectral cube
    
    Returns:
    --------
    list of ssim values
    """
    
    # Get dimensions
    h,w,b = hsi_1.shape
    
    # Initialize list for scores
    ssim_list = []
    
    for k in range(b):
        img1 = MinMax(hsi_1[:,:,k])
        img2 = MinMax(hsi_2[:,:,k])
        ssim_list.append(ssim(img1,img2,data_range=1.0))
    
    return ssim_list



def sam(s1, s2):
    """
    from pydsptools
    Computes the spectral angle mapper between two vectors (in radians).

    Parameters:
        s1: 'numpy array'
        s2: 'numpy array'

    Returns: 'float'
            The angle between vectors s1 and s2 in radians.
    """
    try:
        s1_norm = math.sqrt(np.dot(s1, s1))
        s2_norm = math.sqrt(np.dot(s2, s2))
        sum_s1_s2 = np.dot(s1, s2)
        angle = math.acos(sum_s1_s2 / (s1_norm * s2_norm))
    except ValueError:
        # python math don't like when acos is called with
        # a value very near to 1
        return 0.0
    return angle
    

def sam_perpixel(hsi_1, hsi_2):
    """
    Spectral angle mapper between two hyperspectral images 
    
    Returns:
    --------
    Angles in radian per pixel
    """

    # Dimensions
    assert hsi_1.ndim ==3 and hsi_1.shape == hsi_2.shape
    h, w, b = hsi_1.shape

    # SAM array intialization
    sam_all = np.zeros((h,w))

    for i in range(h):
        for j in range(w):
            sam_all[i,j] = sam(hsi_1[i,j,:],hsi_2[i,j,:])
            
    return sam_all




def make_dark_presentation(
    fig=None,
    text_color="#EAEAEA",
    grid_color="#888888",
    grid_alpha=0.3,
    spine_color="#EAEAEA",
    line_width=None,
    transparent=True,
    black_background=False,
):
    """
    from ChatGPT to export nice figures for powerpoint
    """

    if fig is None:
        fig = plt.gcf()

    # Force draw so tick labels exist
    fig.canvas.draw()

    # --- Figure background ---
    if transparent:
        fig.patch.set_alpha(0)
    elif black_background:
        fig.patch.set_facecolor("black")

    for ax in fig.get_axes():

        if not ax.get_visible():
            continue

        # Background
        if transparent:
            ax.set_facecolor("none")
        elif black_background:
            ax.set_facecolor("black")

        # ---- SPINES ----
        for spine in ax.spines.values():
            spine.set_color(spine_color)

        # ---- TICKS ----
        ax.tick_params(colors=text_color, which='both')

        # Offset text (e.g., 1e-3)
        ax.xaxis.get_offset_text().set_color(text_color)
        ax.yaxis.get_offset_text().set_color(text_color)

        # ---- LABELS + TITLE ----
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

        # ---- GRID (only if already enabled) ----
        xgrid = ax.get_xgridlines()
        ygrid = ax.get_ygridlines()
        
        if any(line.get_visible() for line in xgrid + ygrid):
            for line in xgrid + ygrid:
                line.set_color(grid_color)
                line.set_alpha(grid_alpha)

        # ---- LINES ----
        for line in ax.get_lines():
            if line_width is not None:
                line.set_linewidth(line_width)

        # ---- LEGEND ----
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("none")
            legend.get_frame().set_edgecolor(spine_color)
            for text in legend.get_texts():
                text.set_color(text_color)
            legend.get_title().set_color(text_color)

        # ---- ALL TEXT OBJECTS (annotations etc.) ----
        for text in ax.texts:
            text.set_color(text_color)

    return fig




    



