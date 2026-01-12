"""
Alignment of EDX hyperspectral cube by aligning the EM images using SOFIMA

Ref:
https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_alignment.ipynb


"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils import *
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # change this to your convenience

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorstore as ts
import tempfile
import pathlib

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp
from tqdm.notebook import tqdm
from concurrent import futures
import time
from scipy import interpolate
from scipy.stats import pearsonr
from EDX import *


class sofima_alignment:

    """ class to store the obtained 

        see _compute_flow() for details
    """
    def __init__(self, inv_map, n_align, min_peak_ratio, min_peak_sharpness,
                       max_magnitude, max_deviation, patch_size, stride, pad_remove, box1x):
        
        self.inv_map = inv_map
        self.n_align = n_align
        self.min_peak_ratio = min_peak_ratio
        self.min_peak_sharpness = min_peak_sharpness
        self.max_magnitude= max_magnitude
        self.max_deviation = max_deviation
        self.patch_size = patch_size
        self.stride = stride
        self.pad_remove = pad_remove
        self.box1x = box1x


def _compute_flow(volume, patch_size, stride):
  """ flow estimation. See below
    https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_alignment.ipynb
      
  """
  mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
  flows = []
  prev = volume[..., 0, 0].T.read().result()

  fs = []
  with futures.ThreadPoolExecutor() as tpe:
    # Prefetch the next sections to memory so that we don't have to wait for them
    # to load when the GPU becomes available.
    for z in range(1, volume.shape[2]):
      fs.append(tpe.submit(lambda z=z: volume[..., z, 0].T.read().result()))

    fs = fs[::-1]

    for z in tqdm(range(1, volume.shape[2])):
      curr = fs.pop().result()

      # The batch size is a parameter which impacts the efficiency of the computation (but
      # not its result). It has to be large enough for the computation to fully utilize the
      # available GPU capacity, but small enough so that the batch fits in GPU RAM.
      flows.append(mfc.flow_field(prev, curr, (patch_size, patch_size),
                                  (stride, stride), batch_size=256))
      prev = curr

  return flows


def get_alignment(haadf_stack, 
                  n_align = None,
                  min_peak_ratio=1.1, 
                  min_peak_sharpness=1.1,
                  max_magnitude=0, 
                  max_deviation=0,
                  patch_size = 100,
                  stride = 25,
                  pad_remove = 0):
                  
    """ Get SOFIMA transformation

        Inputs
        ----------
        haadf_stack: size (n_frames, height, width)
                     the way it comes out of the EMD file
        n_align: number of frames to align
        min_peak_ratio: for filtering the flow values (see ref)
        min_peak_sharpness: for filtering the flow values (see ref)
        max_magnitude: for filtering the flow values (see ref) 
        max_deviation: for filtering the flow values (see ref)
        patch_size: XY spatial context used for flow field estimation
        stride: XY distance between centers of adjacent patches.
        
        Output
        ----------
        warped_xyz: the alignment transformation
    """


    # set number of frames to align to all if not set
    if np.isnan(n_align):
        n_align = haadf_stack.shape[0]
    
    
    # Reorder to (h, w, b)- and limit to n_align frames
    data_1x = haadf_stack.transpose(1, 2, 0)[..., None]
    data_1x = data_1x[:,:,:n_align,:]
    
    
    # create a binned (lower-res) version 
    data_2x = np.zeros((1024,1024,n_align,1)) 
    
    for i in range(data_2x.shape[2]):
        data_2x[:,:,i,0] = binning_xy(data_1x[:,:,i,:].squeeze())
    
    
    # convert both sets of images to 0-255, channel wise
    for i in range(data_2x.shape[2]):
        data_1x[:,:,i,0] = Normalize_uint8(data_1x[:,:,i,0]) #,normalize_by=data_1x[:,:,:,0]) #you can normalize by the whole stack
        data_2x[:,:,i,0] = Normalize_uint8(data_2x[:,:,i,0]) #,normalize_by=data_2x[:,:,i,0])
    data_1x = data_1x.astype('uint8')
    data_2x = data_2x.astype('uint8')
    
    
    # Create the TensorStore objects for the stacks of images.
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = pathlib.Path(tmp_root)

        ds1_path = tmp_root/"dataset1"
        ds2_path = tmp_root/"dataset2"

    
    unaligned_1x = ts.open({
        'driver': 'n5',
        'kvstore': {
             'driver': 'file',
             'path': str(ds1_path),
         },
         'metadata': {
             'compression': {
                 'type': 'gzip'
             },
             'dataType': 'uint8',
             'dimensions': data_1x.shape,
             'blockSize': [100, 100, 1, 1],
         },
         'create': True,
         'delete_existing': True,}).result()
    
    unaligned_2x = ts.open({
        'driver': 'n5',
        'kvstore': {
             'driver': 'file',
             'path': str(ds2_path),
         },
         'metadata': {
             'compression': {
                 'type': 'gzip'
             },
             'dataType': 'uint8',
             'dimensions': data_2x.shape,
             'blockSize': [50, 50, 1, 1],
         },
         'create': True,
         'delete_existing': True,}).result()
    
    write_future = unaligned_1x.write(data_1x); write_future.result()
    write_future = unaligned_2x.write(data_2x); write_future.result()
    
    # Estimate flows
    flows1x = np.array(_compute_flow(unaligned_1x, patch_size, stride))
    flows2x = np.array(_compute_flow(unaligned_2x, patch_size, stride))

    # Convert to [channels, z, y, x].
    flows2x = np.transpose(flows2x, [1, 0, 2, 3])
    flows1x = np.transpose(flows1x, [1, 0, 2, 3])

    # Pad to account for the edges of the images where there is insufficient context to estimate flow.
    pad = patch_size // 2 // stride
    flows1x = np.pad(flows1x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)
    flows2x = np.pad(flows2x, [[0, 0], [0, 0], [pad, pad], [pad, pad]], constant_values=np.nan)

    # Remove uncertain flows
    f1 = flow_utils.clean_flow(flows1x, min_peak_ratio=min_peak_ratio, min_peak_sharpness=min_peak_sharpness, max_magnitude=max_magnitude, max_deviation=max_deviation)
    f2 = flow_utils.clean_flow(flows1x, min_peak_ratio=min_peak_ratio, min_peak_sharpness=min_peak_sharpness, max_magnitude=max_magnitude, max_deviation=max_deviation)


    # Interpolate
    f2_hires = np.zeros_like(f1)
    
    scale = 0.5
    oy, ox = np.ogrid[:f2.shape[-2], :f2.shape[-1]]
    oy = oy.ravel() / scale
    ox = ox.ravel() / scale
    
    box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f1.shape[-1], f1.shape[-2], 1))
    box2x = bounding_box.BoundingBox(start=(0, 0, 0), size=(f2.shape[-1], f2.shape[-2], 1))
    
    for z in tqdm(range(f2.shape[1])):
      # Upsample and scale spatial components.
      resampled = map_utils.resample_map(
          f2[:, z:z + 1, ...],  #
          box2x, box1x, 1 / scale, 1)
      f2_hires[:, z:z + 1, ...] = resampled / scale

    final_flow = flow_utils.reconcile_flows((f1, f2_hires), max_gradient=0, max_deviation=20, min_patch_size=400)

    # mesh optimzation
    config = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.01, k=0.1, stride=(stride, stride), num_iters=1000,
                                max_iters=100000, stop_v_max=0.005, dt_max=1000, start_cap=0.01,
                                final_cap=10, prefer_orig_order=True)
    
    solved = [np.zeros_like(final_flow[:, 0:1, ...])]
    origin = jnp.array([0., 0.])
    
    for z in tqdm(range(0, final_flow.shape[1])):
      prev = map_utils.compose_maps_fast(final_flow[:, z:z+1, ...], origin, stride,
                                         solved[-1], origin, stride)
      x = np.zeros_like(solved[0])
      x, e_kin, num_steps = mesh.relax_mesh(x, prev, config)
      x = np.array(x)
      solved.append(x)
    
    solved = np.concatenate(solved, axis=1)

    # Warping
    crop_size = 2048 - 2*pad_remove
    inv_map = map_utils.invert_map(solved, box1x, box1x, stride)

    # output
    out = sofima_alignment(inv_map, n_align, min_peak_ratio, min_peak_sharpness,
                       max_magnitude, max_deviation, patch_size, stride, pad_remove, box1x)

    return out


def apply_alignment_2D(img_stack, alignment, data_type):
    """
    Apply a sofima alignment on a single stack of images

    Parameters:
    -----------
    img_stack: stack of images to apply the alignment to (n_frames, height, width)
    alignment: the alignment object
    data_type: of the input and output

    Returns:
    img_stack_alligned: (height, width, n_frames)
    """
    
    # get values from the alignment object
    n_align = alignment.n_align
    pad_remove = alignment.pad_remove
    stride = alignment.stride
    crop_size = 2048 - 2*pad_remove
    inv_map = alignment.inv_map
    box1x = alignment.box1x
    
    # Reorder to (h, w, b)- and limit to n_align frames
    data_1x = img_stack.transpose(1, 2, 0)[..., None]
    data_1x = data_1x[:,:,:n_align,:]
    

    # convert to 0-255, channel wise
    if str(data_type) == 'uint8': 
        for i in range(data_1x.shape[2]):
            data_1x[:,:,i,0] = Normalize_uint8(data_1x[:,:,i,0]) #,normalize_by=data_1x[:,:,:,0]) #you can normalize by the whole stack
        data_1x = data_1x.astype('uint8')

    
    
    # Create the TensorStore objects for the stacks of images.
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = pathlib.Path(tmp_root)
        ds1_path = tmp_root/"dataset1"

    
    unaligned_1x = ts.open({
        'driver': 'n5',
        'kvstore': {
             'driver': 'file',
             'path': str(ds1_path),
         },
         'metadata': {
             'compression': {
                 'type': 'gzip'
             },
             'dataType': data_type,
             'dimensions': data_1x.shape,
             'blockSize': [100, 100, 1, 1],
         },
         'create': True,
         'delete_existing': True,}).result()

    write_future = unaligned_1x.write(data_1x)
    write_future.result()
    
    # Initialize warped list with cropped frame 0
    warped = [np.transpose(unaligned_1x[pad_remove:2048-pad_remove,pad_remove:2048-pad_remove,0:1,0].read().result(),[2, 1, 0])]
    
    for z in tqdm(range(1, unaligned_1x.shape[2])):
    
        data_box = bounding_box.BoundingBox(start=(0, 0, 0),size=(2048, 2048, 1))
        out_box = bounding_box.BoundingBox(start=(pad_remove, pad_remove, 0),size=(crop_size, crop_size, 1))
        data = np.transpose(unaligned_1x[data_box.start[0]:data_box.end[0],data_box.start[1]:data_box.end[1],z:z+1,0:1].read().result(),[3, 2, 1, 0])
    
        warped_slice = warp.warp_subvolume(data,data_box,inv_map[:, z:z+1, ...],box1x,stride,
            out_box,
            'lanczos',
            parallelism=1
        )[0, ...]
    
        warped.append(warped_slice)
    
    img_stack_aligned = np.transpose(np.concatenate(warped, axis=0), [2, 1, 0])

    return img_stack_aligned
    


def store_unaligned_hsi(emd_path, out_path, n_frames):
    """
    Store temporarily the stack of unaligined
    EDX cubes in order to align them

    Parameters:
    -----------
    emd_path: path to the EMD file containing the EDX data
    n_frames: the number of frames that should be aligned

    out_path: where the unaligned stack of EDX is saved    

     Returns
    -------
    ts.TensorStore
    """

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load first frame
    edx_tmp, _, _ = load_EDX(
        file_path=emd_path,
        first_frame=0,
        last_frame=1,
        sum_frames=True,
        haadf_last_frame=False
    )

    # Bin Z-axis (discard first 96 channels)
    edx_tmp = binning_xyz(edx_tmp[:, :, 96:], dim=[2048, 2048, 250])
    h,w,b = edx_tmp.shape 

    # Allocate full array in (h, w, n_frames, b)
    data_1x = np.zeros((h, w, n_frames, b), dtype=np.float32)

    # Store frame 0
    data_1x[:, :, 0, :] = edx_tmp.astype(np.float32)

    # ---- Load remaining frames ----
    for k in range(1, n_frames):
        edx_tmp, _, _ = load_EDX(
            file_path=emd_path,
            first_frame=k,
            last_frame=k+1,
            sum_frames=True,
            haadf_last_frame=False
        )

        edx_tmp = binning_xyz(edx_tmp[:, :, 96:], dim=[2048, 2048, 250])
        data_1x[:, :, k, :] = edx_tmp.astype(np.float32)

        print(f"Loaded frame {k+1:02d}/{n_frames:02d}", end="\r")

    print("\nAll frames loaded.")

    # ---- Write TensorStore ----
    store = ts.open({
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": str(out_path),
        },
        "metadata": {
            "compression": {"type": "gzip"},
            "dataType": "float32",
            "dimensions": data_1x.shape,  # (h, w, n_frames, b)
            "blockSize": [64, 64, 1, b],  
        },
        "create": True,
        "delete_existing": True,
    }).result()

    store.write(data_1x).result()
    print(f"Saved unaligned hyperspectral stack to: {out_path}")

    return store


def store_unaligned_hsi_alt(emd_path, out_path, n_frames):  # improved by GPT

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Load one frame to get shape ----
    edx_tmp, _, _ = load_EDX(
        file_path=emd_path,
        first_frame=0,
        last_frame=1,
        sum_frames=True,
        haadf_last_frame=False
    )

    edx_tmp = binning_xyz(edx_tmp[:, :, 96:], dim=[2048, 2048, 250])
    h, w, b = edx_tmp.shape

    # ---- Create TensorStore on disk ----
    store = ts.open({
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": str(out_path),
        },
        "metadata": {
            "compression": {"type": "gzip"},
            "dataType": "float32",
            "dimensions": [h, w, n_frames, b],
            "blockSize": [64, 64, 1, b],  # frame-aligned chunks
        },
        "create": True,
        "delete_existing": True,
    }).result()

    # ---- Write frame 0 ----
    store[:, :, 0, :].write(edx_tmp.astype(np.float32)).result()

    # ---- Stream remaining frames ----
    for k in range(1, n_frames):
        edx_tmp, _, _ = load_EDX(
            file_path=emd_path,
            first_frame=k,
            last_frame=k+1,
            sum_frames=True,
            haadf_last_frame=False
        )

        edx_tmp = binning_xyz(edx_tmp[:, :, 96:], dim=[2048, 2048, 250])

        store[:, :, k, :].write(edx_tmp.astype(np.float32)).result()

        print(f"Loaded frame {k+1:02d}/{n_frames:02d}", end="\r")

    print("\nAll frames stored.")
    return store


def apply_alignment_3D(hsi_stack_loc_path, alignment, data_type):   
    
    """
    Apply a sofima alignment on a stack of HSIs

    Parameters:
    -----------
    hsi_stack_loc_path: location to the stack of TensorStore of 
                        HSI to apply the alignment to (h, w, n_frames, b)
                        
    alignment: the alignment object
    data_type: of the input and output

    Returns: the sum of the aligned HSIs (h, w, b)
    """

    # load the stack
    store = ts.open({
        "driver": "n5",
        "kvstore": {
            "driver": "file",
            "path": hsi_stack_loc_path,
        },
        "open": True
    }).result()
        
    # get dimensions
    h, w, n_align, b = store.shape

    # Initialize a summed (and aligned frame)
    pad_remove = alignment.pad_remove
    crop_size_h = h - 2*pad_remove
    crop_size_w = w - 2*pad_remove
    hsi_summed_aligned = np.zeros((crop_size_h,crop_size_w,b),dtype=data_type)

    # Align channel by channel using the sofima alignment and add
    for k in range(b):
        # a single stack of images to be aligned
        img_stack = store[:,:,:,k].read().result()
        img_stack_aligned = apply_alignment_2D(np.transpose(img_stack, [2, 0, 1]), alignment, data_type)
        img_stack_aligned_summed = img_stack_aligned.sum(axis=2)
        hsi_summed_aligned[:,:,k] = hsi_summed_aligned[:,:,k]+img_stack_aligned_summed
        print("Channel %03d out of %03d has been aligned" % (k+1,b))

    return hsi_summed_aligned




def eval_alignment(img_stack_unaligned, img_stack_aligned, metric=None):
   
    """
    Evaluate a (dis)similarity metric between first frame and subsequent 
    ones, before and after alignment

    Parameters:
    -----------
    img_stack_unaligned: the imack stack before alignment (h, w, frames or depth etc) 
    img_stack_aligned: the image stack after alignment (h, w, frames or depth etc)
    they should have the same dtype
    metric: the evaluation metric

    Returns:
    --------
    metric_before: metric list before alignment
    metric_after: metric list after alignment
    """

    # get dimensions
    h, w, n_align = img_stack_unaligned.shape

    # Prepare metric function
    if metric is None or metric == 'pcc':
        def metric_func(x, y):
            return pearsonr(x, y)[0]
    else:
        # Callable passed by user
        metric_func = metric

    metric_before_list = []
    metric_after_list = []

    # Reference frame = frame 0
    ref_before = img_stack_unaligned[:, :, 0].ravel()
    ref_after  = img_stack_aligned[:, :, 0].ravel()

    for i in range(n_align):

        curr_before = img_stack_unaligned[:, :, i].ravel()
        curr_after  = img_stack_aligned[:, :, i].ravel()

        metric_before_list.append(metric_func(ref_before, curr_before))
        metric_after_list.append(metric_func(ref_after, curr_after))

    return metric_before_list, metric_after_list
    
        
    

    
    