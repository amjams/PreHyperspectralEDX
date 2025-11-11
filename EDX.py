"""The EM-EDX class

   and the corresponding preprocessing methods.
"""

# Authors: Ahmad Alsahaf
import sys, os
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
import pandas as pd
from utils import *
from bm3d import bm3d



class EM_EDX:
    """
    A class to define a tile of co-registered HAADF and EDX data with provenance tracking.
    """

    def __init__(self, haadf, EDX, xray_energies, metadata=None):
        self.haadf = haadf
        self.EDX = EDX
        self.xray_energies = xray_energies
        self.metadata = metadata if metadata else {}
        self.processing_history = []

    @property
    def haadf_dim(self):
        return self.haadf.shape

    @property
    def EDX_dim(self):
        return self.EDX.shape
    
    @property
    def EDX_2D(self):
        h,w,b = self.EDX_dim
        return self.EDX.reshape((h*w,b))

    def log_step(self, operation, parameters=None, notes=None):
        """Log a preprocessing step."""
        self.processing_history.append({
            "operation": operation,
            "parameters": parameters or {},
            "haadf size": self.haadf_dim,
            "EDX size": self.EDX_dim,
            "notes": notes or None,
        })

    def apply(self, method_name, parameters=None, notes=None, copy_instance=False):
        """
        Apply a preprocessing method (by name) to this dataset, optionally copying first.

        Parameters
        ----------
        method_name : str
            Name of the method to apply
        parameters : dict, optional
            Parameters passed to the method.
        notes : str, optional
            Free-form text for provenance.
        copy_instance : bool, default False
            Whether to apply on a deepcopy or in place.

        Returns
        -------
        EM_EDX
            The modified or new dataset instance.
        """
        parameters = parameters or {}

        # Decide whether to work on a copy
        obj = copy.deepcopy(self) if copy_instance else self

        # Get the method and apply
        if not hasattr(obj, method_name):
            raise AttributeError(f"Method '{method_name}' not found in EM_EDX")
        method = getattr(obj, method_name)
        result = method(**parameters)
        if not isinstance(result, EM_EDX):
            raise TypeError(f"Method {method_name} must return an EM_EDX instance.")

        # Log the operation on the modified object
        result.log_step(method_name, parameters, notes)
        return result


    # Preprocessing methods
    def crop(self, crop_idx=None):
        """
        Jointly crop the HAADF, EDX, and xray_energies arrays.
        
        Parameters
        ----------
        crop_idx : tuple of slices, ints, or None
            A tuple specifying the crop region.
            For example:
            (slice(y0, y1), slice(x0, x1)) for spatial
            (slice(y0, y1), slice(x0, x1), slice(z0, z1)) for spatial spectral.

            or ints:
            (y0, y1, x0, x1[, z0, z1]) defining crop bounds.
        """
        if crop_idx is None:
            return self
        
        elif isinstance(crop_idx[0], slice):
            y_slice, x_slice = crop_idx[:2]
            self.haadf = self.haadf[y_slice, x_slice]
            self.EDX = self.EDX[crop_idx]
            self.xray_energies = self.xray_energies[crop_idx[2]] if len(crop_idx) == 3 else self.xray_energies

        else:
            if len(crop_idx) == 4:
                y0, y1, x0, x1 = crop_idx
                self.haadf = self.haadf[y0:y1, x0:x1]
                self.EDX = self.EDX[y0:y1, x0:x1, :]
            elif len(crop_idx) == 6:
                y0, y1, x0, x1, z0, z1 = crop_idx
                self.haadf = self.haadf[y0:y1, x0:x1]
                self.EDX = self.EDX[y0:y1, x0:x1, z0:z1]
                self.xray_energies = self.xray_energies[z0:z1]
            else:
                raise ValueError("crop_idx must be a 4-tuple or 6-tuple of ints.")
        return self


    def binning(self, dim = None):
        """
        Jointly bin the HAADF, EDX, and xray_energies arrays.
        according to the given new dimensions.
        
        Parameters
        ----------
        dim : tuple of ints, specifying the new x,y,b dimensions.
            A tuple specifying the crop region.
        """
        if dim is None:
            return self

        if len(dim) != 3:
            raise ValueError('dim must be a 3-tuple: x,y,b new dimensions.')    
            
        if any(self.EDX_dim[i] % dim[i] != 0 for i in range(3)):
            raise ValueError("Ensure old dims are divisible by new dims.")
            
        
        # original and new dimensions
        old_y, old_x, old_b = self.EDX_dim
        new_y, new_x, new_b = dim
    
        # binning factors
        fy, fx, fb = old_y // new_y, old_x // new_x, old_b // new_b
    
        # HAADF 
        self.haadf = self.haadf.reshape(new_y, fy, new_x, fx)
        self.haadf = self.haadf.mean(axis=(1, 3))
    
        # EDX 
        self.EDX = self.EDX.reshape(new_y, fy, new_x, fx, new_b, fb)
        self.EDX = self.EDX.mean(axis=(1, 3, 5))
    
        # xray energies 
        self.xray_energies = self.xray_energies.reshape(new_b, fb)
        self.xray_energies = self.xray_energies.mean(axis=-1)
            
        return self

    def MeanFilterEDX(self,kernel_size=3):
        """
        Apply a mean filter per band on the EDX cube
        
        Parameters
        ----------
        kernel_size: size (width/height) of the kernel
        """

        # apply a mean filter per band
        for k in range(self.EDX_dim[2]):
            self.EDX[:,:,k] = mean_filter(self.EDX[:,:,k],kernel_size=kernel_size)
            
        return self

    def MinMaxEDX(self, bandwise = False):
        """
        min-max normalize the EDX datacube.
        
        Parameters
        ----------
        bandwise: if True, normalize per band.
        """

        b = self.EDX_dim[2]

        if bandwise:
            for k in range(b):
                self.EDX[:,:,k] = MinMax(self.EDX[:,:,k])
        else:
            self.EDX = MinMax(self.EDX)    
        return self

    def FalseColor(self,bands=[4,25,28]):
        ## return a false color of three selected bands
        r = Normalize_uint8(self.EDX[:,:,bands[0]])
        g = Normalize_uint8(self.EDX[:,:,bands[1]])
        b = Normalize_uint8(self.EDX[:,:,bands[2]])
        
        return cv.merge([r,g,b])

    def PCA_bm3d(self, k=10, sigma=0.1, zscore=False):
        """
        Denoise with PCA + BM3D
        
        
        Parameters
        ----------
        k: first k-components which are not denoised
        sigma: std of the noise (parameter for bm3d)
        """
        h, w, b = self.EDX_dim
        pca_model = PCA()
        pca_model.fit(self.EDX_2D)
        pca_scores = pca_model.transform(self.EDX_2D)
        pca_scores_denoised = pca_scores.copy()

        # denoise channels after p with bm3d
        for i in range(k):
            if i<k:
                denoise_channel = pca_scores[:,i].reshape((h,w))
                denoise_channel = bm3d(denoise_channel, sigma) 
                pca_scores_denoised[:,i] =  denoise_channel.reshape((h*w,))

        hsi_denoised_2D = pca_model.inverse_transform(pca_scores_denoised)
        self.EDX = hsi_denoised_2D.reshape((h,w,b))
        return self


    def summary(self):
        """Return a pandas DataFrame summarizing the preprocessing history."""
        if not self.processing_history:
            print("No preprocessing steps recorded.")
            return pd.DataFrame(columns=["operation", "parameters", "notes"])

        df = pd.DataFrame(self.processing_history)
        # Make parameters easier to read
        df["parameters"] = df["parameters"].apply(
            lambda p: ", ".join(f"{k}={v}" for k, v in p.items()) if p else ""
        )
        return df

    def __repr__(self):
        return f"<EM_EDX | {len(self.processing_history)} steps logged>"


