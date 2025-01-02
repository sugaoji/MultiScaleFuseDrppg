import torch
import os
import numpy as np
import random
import glob
from scipy.io import loadmat
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import config


class DataLoader(Dataset):

    def __init__(self, maps_path, transforms=None, random_mask=None):
        st_map_path = []
        for file in maps_path:
            st_map_path.extend(glob.glob(file+"/*"))
        self.st_map_path = st_map_path
        self.transforms=transforms
        self.random_mask=random_mask

    def __len__(self):
        return len(self.st_map_path)

    def __getitem__(self, item):
        root_path = self.st_map_path[item]
        path = root_path.split('/')[-2]+'-'+root_path.split('/')[-1]
        rgb_img_path = os.path.join(root_path, "img_rgb.png")
        yuv_img_path = os.path.join(root_path, "img_yuv.png")
        map1 = Image.open(rgb_img_path).convert("RGB")
        map2 = Image.open(yuv_img_path).convert("RGB")
        if self.transforms:
            feature_map = np.concatenate((np.array(map1), np.array(map2)), axis=2)
#             print(feature_map.shape)
#             feature_map = Image.fromarray(feature_map)
            feature_map = self.transforms(feature_map)
#             map1 = self.transforms(map1)
#             map2 = self.transforms(map2)
#         feature_map = torch.cat((map1, map2), dim=0)
        # gt_HR = loadmat(os.path.join(root_path, "gt_HR.mat"))["gt_HR_temp"].squeeze(0)
        hr = loadmat(os.path.join(root_path, "hr.mat"))["hr"].squeeze(0)
        bvp = loadmat(os.path.join(root_path, "filter_bvp.mat"))["bvp"].squeeze(0)
        bvp = (bvp-np.mean(bvp))/np.std(bvp)
        # gt_HR = torch.tensor(gt_HR, dtype=torch.float32)
        hr = torch.tensor(hr, dtype=torch.float32)
        bvp = torch.tensor(bvp, dtype=torch.float32)

        return {"st_maps": feature_map, "gt_HR": hr, "bvp": bvp, "path": path}


if __name__ == "__main__":
    pass