"""The EM-EDX class


And preprocessing methods
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

    def log_step(self, operation, parameters=None, notes=None):
        """Log a preprocessing step."""
        self.processing_history.append({
            "operation": operation,
            "parameters": parameters or {},
            "notes": notes or None,
        })

    def apply(self, method_name, parameters=None, notes=None, copy_instance=False):
        """
        Apply a preprocessing method (by name) to this dataset, optionally copying first.

        Parameters
        ----------
        method_name : str
            Name of the method to apply (e.g., "crop_haadf").
        parameters : dict, optional
            Parameters passed to the method.
        notes : str, optional
            Free-form text for provenance.
        copy_instance : bool, default False
            Whether to apply on a deepcopy (like sklearn transformers) or in place.

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

    # ------------------------
    # Example preprocessing methods (feeling cute, might delete later, idk)
    # ------------------------
    def crop_haadf(self, crop_size=100):
        """Crop the HAADF image."""
        self.haadf = self.haadf[:crop_size, :crop_size]
        return self

    def crop_EDX(self, crop_size=100):
        """Crop the EDX data."""
        self.EDX = self.EDX[:crop_size, :crop_size, :]
        return self

    # ------------------------
    # Real preprocessing methods
    # ------------------------

    def __repr__(self):
        return f"<EM_EDX | {len(self.processing_history)} steps logged>"


