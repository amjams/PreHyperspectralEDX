
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
from EDX import *
from utils import *
from utils_sofima import *
import hyperspy.api as hs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from importlib import reload
from sklearn.preprocessing import MinMaxScaler

from skimage import data, img_as_float, feature
import tensorstore as ts
from scipy.stats import pearsonr
from datetime import datetime
import warnings
import pickle
from datetime import datetime

from itertools import repeat
import random
import logging


# Directory containing EMD files 
emd_dir = "/Volumes/Microscopy4/qdotEDX/EMD"

# Paths to the EMD files
#print(os.listdir(emd_dir))

file_paths =[]
for x in os.listdir(emd_dir):
    if x.endswith('.emd'):
        file_paths.append(os.path.join(emd_dir,x))



for file_path in file_paths:
    # start timer
    start = datetime.now()

    # Tile stamp and file naming suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_frames = 20

    # file name
    fname = os.path.splitext(os.path.basename(file_path))[0]
    suffix = f'{fname}_{num_frames:2d}frames'

    # Load HAADF stack
    EXD_summed_unaligned, haadf_stack, _ = load_EDX(file_path, first_frame=0, last_frame=num_frames, sum_frames=True, haadf_last_frame=False)


    # alignment object settings
    sof_obj = get_alignment(haadf_stack, 
                    n_align = num_frames,
                    min_peak_ratio=1.1, 
                    min_peak_sharpness=1.1,
                    max_magnitude=0, 
                    max_deviation=0,
                    patch_size = 100,
                    stride = 25,
                    pad_remove = 50,
                    align_to_zero = True)


    # Apply to haadf
    haadf_stack_aligned = apply_alignment_2D(haadf_stack, sof_obj, 'uint8')


    # Evaluate
    # making sure the stacks have matched dimensions
    pad_remove = sof_obj.pad_remove
    haadf_stack = np.transpose(haadf_stack,[1,2,0])[pad_remove:2048-pad_remove,pad_remove:2048-pad_remove, :num_frames].astype('uint8')

    print(haadf_stack.shape, haadf_stack_aligned.shape)
    print(haadf_stack.dtype, haadf_stack_aligned.dtype)


    pcc_before, pcc_after = eval_alignment(haadf_stack, haadf_stack_aligned)
    print('Pearson coeffients before and after: ', np.mean(pcc_before), np.mean(pcc_after))


    # Visualize and save
    zoom = (slice(1300,1750),slice(1300,1750))

    f, ax = plt.subplots(2,3,figsize=(15,10))
    ax[0][0].imshow(haadf_stack[:,:,:100].sum(axis=2),cmap='gray_r')
    ax[0][1].imshow(haadf_stack_aligned[:,:,:100].sum(axis=2),cmap='gray_r')
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

    #plt.close()
    make_dark_presentation(f,text_color='white', line_width=2.5, transparent=True)      
    plt.savefig(f"/Volumes/Microscopy4/qdotEDX/Alignment_data/Alignment_illustration/{suffix}.png", dpi=300, transparent=True) 


    # Save both stacks
    stacks = [haadf_stack, haadf_stack_aligned]
    np.savez(f"/Volumes/Microscopy4/qdotEDX/Alignment_data/Full_haadf_stacks/{suffix}.npz", haadf_stack=haadf_stack, 
             haadf_stack_aligned=haadf_stack_aligned, pcc_before=np.array(pcc_before), pcc_after=np.array(pcc_after))


    # Save alignment object
    with open(f"/Volumes/Microscopy4/qdotEDX/Alignment_data/Sofima_objects/{suffix}.pkl", "wb") as f:
        pickle.dump(sof_obj, f)


    # Store unaligned (tensorstore)
    logging.getLogger("rsciio.emd").setLevel(logging.ERROR)
    tmp = store_unaligned_hsi_alt(file_path,f"/Volumes/Microscopy4/tmp/{suffix}",n_frames=num_frames)


    # Load from EMD
    edx_unaligned, haadf, xray_energies = load_EDX(file_path, first_frame=0, last_frame=num_frames, sum_frames=True, haadf_last_frame= False)


    # Create tile EM/EDX objects for denoising and alignment
    tile_unaligned = EM_EDX(haadf[0,:,:], edx_unaligned, xray_energies)
    tile_unaligned.apply("crop", parameters={"crop_idx": (slice(None), slice(None), slice(96, 4096))})
    tile_unaligned.apply("binning", parameters={"dim": (2048, 2048, 250)})

    # Algined
    pad_remove = sof_obj.pad_remove
    tile_aligned = EM_EDX(haadf[0,:,:], edx_unaligned, xray_energies)
    tile_aligned.apply("crop", parameters={"crop_idx": (slice(None), slice(None), slice(96, 4096))})
    tile_aligned.apply("binning", parameters={"dim": (2048, 2048, 250)})


    # Apply the alignment
    tile_aligned.apply("sofima_align", parameters={"hsi_stack_path": f"/Volumes/Microscopy4/tmp/{suffix}", "alignment": sof_obj, "data_type": "float32",
                                        "save_aligned": False, "hsi_stack_aligned_path": f"tmp/aligned_hsi_{suffix}"})   

    # Save (these are at full spatial resolution; not yet binned or mean filtered)
    # save to memory

    with open(f"/Volumes/Microscopy4/qdotEDX/Preprocessed_EMEDX/unaligned/unaligned_{suffix}.pkl", "wb") as f:
        pickle.dump(tile_unaligned, f)


    with open(f"/Volumes/Microscopy4/qdotEDX/Preprocessed_EMEDX/aligned/aligned_{suffix}.pkl", "wb") as f:
        pickle.dump(tile_aligned, f)


    # Prepare and apply NGMeet both 

    # Unaligned and denoised
    tile_unaligned.apply("crop", parameters = {"crop_idx": (slice(50,tile_unaligned.EDX_dim[0]-50),slice(50,tile_unaligned.EDX_dim[1]-50),slice(None,None,None))})
    tile_unaligned.apply("binning", parameters={"dim": (974,974,250)})
    tile_unaligned.apply("MeanFilterEDX", parameters={"kernel_size": 3})

    sigma = 0.4
    tile_unaligned.apply("NGMeet_matlab", parameters = {"sigma": sigma, "nEM": 10, "nIter": 2},copy_instance=False)

    with open(f"/Volumes/Microscopy4/qdotEDX/Preprocessed_EMEDX/unaligned_denoised/unaligned_denoised_{suffix}.pkl", "wb") as f:
        pickle.dump(tile_unaligned, f)


    # Aligned and denoised
    tile_aligned.apply("crop", parameters = {"crop_idx": (slice(50,tile_aligned.EDX_dim[0]-50),slice(50,tile_aligned.EDX_dim[1]-50),slice(None,None,None))})
    tile_aligned.apply("binning", parameters={"dim": (974,974,250)})
    tile_aligned.apply("MeanFilterEDX", parameters={"kernel_size": 3})

    sigma = 0.4
    tile_aligned.apply("NGMeet_matlab", parameters = {"sigma": sigma, "nEM": 10, "nIter": 2},copy_instance=False)

    with open(f"/Volumes/Microscopy4/qdotEDX/Preprocessed_EMEDX/aligned_denoised/aligned_denoised_{suffix}.pkl", "wb") as f:
        pickle.dump(tile_aligned, f)


    end = datetime.now()
    duration = end - start
    duration_in_minutes_and_seconds = str(duration).split('.')[0][2:]

    print(f"Elapsed time for preprocessing {fname}: {duration_in_minutes_and_seconds}")


