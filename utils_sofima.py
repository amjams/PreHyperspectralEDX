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


#class sofima_transform:
#    self.


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
    
    
    # Reorder to (h, w, b)- and limit to 20 frames
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
    unaligned_1x = ts.open({
        'driver': 'n5',
        'kvstore': {
             'driver': 'file',
             'path': 'tmp/dataset1/',
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
             'path': 'tmp/dataset2/',
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
    
    # Initialize warped list with cropped slice 0
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
    
    warped_xyz = np.transpose(np.concatenate(warped, axis=0), [2, 1, 0])
    
    return warped_xyz


    