import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # change this to your convenience
import jax
print(jax.default_backend()) 

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from connectomics.common import bounding_box
from sofima import flow_field
from sofima import flow_utils
from sofima import map_utils
from sofima import mesh
from sofima import warp
from tqdm.notebook import tqdm
from concurrent import futures
import time


# number of channels to align
n_channels = 20


# Reorder to (2048, 2048, 100)- and limit to 20 frames
data_1x = haadf.transpose(1, 2, 0)[..., None]
data_1x = data_1x[:,:,:n_channels,:]


# create a binned version (also to match the unaligned_2x in the demo) 
data_2x = np.zeros((1024,1024,n_channels,1)) 

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

write_future = unaligned_1x.write(data_1x)
write_future.result()
write_future = unaligned_2x.write(data_2x)
write_future.result()