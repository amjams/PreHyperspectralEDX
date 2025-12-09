import torch
from torch.utils.data import Dataset
import tensorstore as ts
import numpy as np

class HyperspectralPatchDataset(Dataset):
    def __init__(self, store_path, patch_size=(64,64), bands=None, n_input_frames=10, n_patches=1000):
        self.store = ts.open({
            "driver": "n5",
            "kvstore": {
                "driver": "file",
                "path": store_path,
            },
            "open": True
        }).result()

        
        self.h, self.w, self.n_frames, self.b = self.store.shape
        self.patch_size = patch_size
        self.n_input_frames = n_input_frames
        self.n_patches = n_patches
        self.bands = bands
        
    def __len__(self):
        return self.n_patches
    
    def __getitem__(self, idx):
        h_t, w_t = self.patch_size
        
        top = np.random.randint(0, self.h - h_t + 1)
        left = np.random.randint(0, self.w - w_t + 1)
        
        frame_idx = np.random.permutation(self.n_frames)
        input_idx = frame_idx[:self.n_input_frames]
        output_idx = frame_idx[self.n_input_frames:]

        if self.bands is None: # use all the HSI
            patch = self.store[top:top+h_t, left:left+w_t, :, :].read().result()
        else:
            patch = self.store[top:top+h_t, left:left+w_t, :, self.bands].read().result()
        patch = np.nan_to_num(patch, nan=0.0)

        
        input_patch = patch[:, :, input_idx, :].sum(axis=2)
        output_patch = patch[:, :, output_idx, :].sum(axis=2)

        
        input_patch = torch.from_numpy(input_patch).float().permute(2,0,1)   # (b, h_t, w_t)
        output_patch = torch.from_numpy(output_patch).float().permute(2,0,1)
        
        return input_patch, output_patch

