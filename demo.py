# Creating and tested the EM-EDX class: creating an object class from the EDX and HAADF data that's in the EMD file. 

import sys, os
from utils import *
from utils_sofima import *
from EDX import *
import numpy as np
import hyperspy.api as hs
import os
from datetime import datetime 
import pickle
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import gc



# load data
file_path = "/scratch/p276451/irodsToHabrok_test/0001 - 2025-284b 12000 x.emd"  # 20 frames max for this file
EDX, haadf_stack, xray_energies = load_EDX(file_path, first_frame=0, last_frame=20,sum_frames=True, haadf_last_frame=False)


# create an out dictory with the name of the EMD file and the current date and time
output_dir = "/scratch/p276451/EM_EDX_output/" + os.path.basename(file_path).split('.')[0] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Multiple steps
# load show dimensions
haadf = haadf_stack[0,:,:]
tile = EM_EDX(haadf, EDX, xray_energies)
print(tile.EDX_dim)

# preprocess
tile.apply("crop", parameters={"crop_idx": (slice(None),slice(None),slice(96,4096))})
tile.apply("binning", parameters={"dim": (1024,1024,250)})
tile.apply("MeanFilterEDX", parameters={"kernel_size": 3})
print(tile.summary())


# visualize the haadf and a false-color of NPS maps and save
nps = tile.FalseColor()
f, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(1-tile.haadf,cmap='gray')
ax[1].imshow(nps)
#plt.show()
make_dark_presentation(f,text_color='white', line_width=2.5, transparent=True)
plt.savefig(output_dir + "/haadf_NPS_after_binning_meanfiltering.png", dpi=300, transparent=True)


####### SOFIMA ALIGNMENT ########
# load and preprocess
num_frames = 20
sof_obj = get_alignment(haadf_stack, 
                  n_align = num_frames,
                  min_peak_ratio=1.1, 
                  min_peak_sharpness=1.1,
                  max_magnitude=0, 
                  max_deviation=0,
                  patch_size = 100,
                  stride = 25,
                  pad_remove = 50,
                  tmp_dir= output_dir, 
                  align_to_zero = True)

# Apply the alignment on the HAADF stack
haadf_stack_aligned = apply_alignment_2D(haadf_stack, sof_obj, 'uint8', tmp_dir= output_dir)


# Ensure the stacks have matched dimensions
pad_remove = sof_obj.pad_remove
haadf_stack = np.transpose(haadf_stack,[1,2,0])[pad_remove:2048-pad_remove,pad_remove:2048-pad_remove, :num_frames].astype('uint8')

# Evaluate the alignment
pcc_before, pcc_after = eval_alignment(haadf_stack, haadf_stack_aligned)
print('Pearson coeffients before and after: ', np.mean(pcc_before), np.mean(pcc_after))


zoom = (slice(1300,1750),slice(1300,1750))

f, ax = plt.subplots(2,3,figsize=(15,10))
ax[0][0].imshow(haadf_stack[:,:,:num_frames].sum(axis=2),cmap='gray_r')
ax[0][1].imshow(haadf_stack_aligned[:,:,:num_frames].sum(axis=2),cmap='gray_r')
ax[0][2].imshow(haadf_stack[:,:,-1],cmap='gray_r')

ax[0][0].set_title('Unaligned',fontsize=20)
ax[0][1].set_title('Aligned',fontsize=20)
ax[0][2].set_title('Single frame',fontsize=20)

ax[1][0].imshow(haadf_stack.sum(axis=2)[zoom],cmap='gray_r')
ax[1][1].imshow(haadf_stack_aligned.sum(axis=2)[zoom],cmap='gray_r')
ax[1][2].imshow(haadf_stack[:,:,-1][zoom],cmap='gray_r')


 # Extract rectangle from zoom slices
ys, xs = zoom
y0 = ys.start or 0
y1 = ys.stop or haadf_stack[:,:,:20].sum(axis=2).shape[0]
x0 = xs.start or 0
x1 = xs.stop or haadf_stack_aligned[:,:,:20].sum(axis=2).shape[1]


ax[0][0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                      linewidth=1, edgecolor='white', facecolor='none'))
ax[0][1].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                      linewidth=1, edgecolor='white', facecolor='none'))
ax[0][2].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                      linewidth=1, edgecolor='white', facecolor='none'))

for a in ax.ravel():
    a.axis('off')

make_dark_presentation(f,text_color='white', line_width=2.5, transparent=True)
plt.savefig(output_dir + "/visualization_of_aligning_haadf.png", dpi=300, transparent=True)



# Save the alignment object
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = output_dir + "/sof_object.pkl"
with open(save_path, "wb") as f:
    pickle.dump(sof_obj, f)


# Save the EM_EDX object to a file
save_path = output_dir + "/EM_EDX_object_binned_meanfiltered.pkl"
with open(save_path, "wb") as f:
    pickle.dump(tile, f)



# Saving with TensorStore the unaligned EDX frames 
import logging
logging.getLogger("rsciio.emd").setLevel(logging.ERROR)

save_path = output_dir + "/tmp/unaligned_hsi"
tmp = store_unaligned_hsi_alt(file_path, save_path, n_frames=num_frames)
print("The unaligned HSI has been stored in: %s " % save_path)


# Memory management
del tile, haadf_stack_aligned, haadf_stack
gc.collect()

## Create two EMD objects, one aligned, one not

# 1) load data again from EMD
edx_unaligned, haadf, xray_energies = load_EDX(file_path, first_frame=0, last_frame=num_frames, sum_frames=True, haadf_last_frame= False)

# 2) Unaligned EM-EDX object, binned
tile_1 = EM_EDX(haadf[0,:,:], edx_unaligned, xray_energies)
tile_1.apply("crop", parameters={"crop_idx": (slice(None), slice(None), slice(96, 4096))})
tile_1.apply("binning", parameters={"dim": (2048, 2048, 250)})
print('Unaligned object created')

# 3) Aligned
pad_remove = sof_obj.pad_remove
tile_2 = EM_EDX(haadf[0,:,:], edx_unaligned, xray_energies)
tile_2.apply("crop", parameters={"crop_idx": (slice(None), slice(None), slice(96, 4096))})
tile_2.apply("binning", parameters={"dim": (2048, 2048, 250)})

# Align
tile_2.apply("sofima_align", 
             parameters={"hsi_stack_path": "tmp/unaligned_hsi20230930 0546 12000 x 2023-146_20frames_align2zero",
                          "alignment": sof_obj, 
                          "data_type": "float32",
                          "save_aligned": False, 
                          "hsi_stack_aligned_path": None})   


# Save the aligned EM-EDX tile
# Save the EM_EDX object to a file
save_path = output_dir + "/EM_EDX_object_SpectrallyBinned_SofimaAligned.pkl"
with open(save_path, "wb") as f:
    pickle.dump(tile_2, f)