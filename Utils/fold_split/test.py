import glob
import os
from sklearn import model_selection
import pandas as pd
import numpy as np
import torch
import random
import scipy.io
import sys

fold_files = glob.glob("/qianwei/NiuXueSongforHR/NLnetforHR/src_code/Utils/fold_split/*.mat")
for fold in fold_files:
    name = fold.split('/')[-1].split('.')[0]
    fold_data = scipy.io.loadmat(fold)
    print(fold_data[name])

data = scipy.io.loadmat("/qianwei/NiuXueSongforHR/NLnetforHR/src_code/Utils/fold_split/fold3.mat")
data = data['fold3'].squeeze(0)
count = 0
for x in data:
    path = glob.glob(f'/qianwei/dataset/preVIPL/VIPL_video/p{x}_*')
    path = [x for x in path if 'source4' not in x]
    count += len(path)
print(count)

     
