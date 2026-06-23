# Creating and tested the EM-EDX class: creating an object class from the EDX and HAADF data that's in the EMD file. 

import sys, os
from utils import *
from EDX import *
import numpy as np
import hyperspy.api as hs
import os
from datetime import datetime 
import pickle



# load data
file_path = "/scratch/p276451/irodsToHabrok_test/0001 - 2025-284b 12000 x.emd"  # 20 frames max for this file
EDX, haadf, xray_energies = load_EDX(file_path, first_frame=0, last_frame=20,sum_frames=True)  

# create an out dictory with the name of the EMD file and the current date and time
output_dir = "/scratch/p276451/EM_EDX_output/" + os.path.basename(file_path).split('.')[0] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Multiple steps
# load show dimensions
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
plt.savefig(output_dir + "/test.png", dpi=300, transparent=True)



# Save the EM_EDX object to a file
save_path = output_dir + "/EM_EDX_object_binned_meanfiltered.pkl"
with open(save_path, "wb") as f:
    pickle.dump(tile, f)


