"""The EM-EDX class

   and the corresponding preprocessing methods.
"""

# Authors: Ahmad Alsahaf

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


    # Summarize the processing history in a pandas dataframe
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


