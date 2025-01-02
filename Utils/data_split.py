import glob
import os
from sklearn import model_selection
import pandas as pd
import numpy as np
import torch
import random
import scipy.io
import sys 
sys.path.append("..") 
import config


def SplitforMaps(file_list, n_splits=5):
    """
    :param file_list: maps path list [./p*_v*_source*,..]
    :param n_splits: 5
    :return: 两个5折list 每个元素是训练/验证 数据路径：..\p47_v5_source2\1
    """
    kf = model_selection.KFold(n_splits=n_splits, shuffle=False)
    file_list = [x for x in file_list if os.listdir(x)]
    train_ls = []
    test_ls = []
    for train_idx, test_idx in kf.split(file_list):
        train_dir = [file_list[x] for x in train_idx]
        test_dir = [file_list[x] for x in test_idx]
        train_files = []
        test_files = []
        for path in train_dir:
            train_files.extend(glob.glob(os.path.join(path, '*')))
        train_ls.append(train_files)
        for path in test_dir:
            test_files.extend(glob.glob(os.path.join(path, '*')))
        test_ls.append(test_files)

    return train_ls, test_ls


def SplitforVideos(file_list, n_splits=4):
    """
    :param file_list: maps path list [./p*_v*_source*,..]
    :param n_splits: 5
    :return: 两个5折list 每个元素是训练/验证 数据路径：..\p47_v5_source2
    """
    kf = model_selection.KFold(n_splits=n_splits, shuffle=False)
    file_list = [x for x in file_list if os.listdir(x)]
    train_ls = []
    test_ls = []
    for train_idx, test_idx in kf.split(file_list):
        train_dir = [file_list[x] for x in train_idx]
        test_dir = [file_list[x] for x in test_idx]

        train_ls.append(train_dir)
        test_ls.append(test_dir)

    return train_ls, test_ls

def SplitforSubject(file_list, name, n_splits=5):
    """
    :param file_list: 每条video的dir
    :param n_splits:
    :return: 按subject划分的train和test
    """
    kf = model_selection.KFold(n_splits=n_splits, shuffle=True)
    file_list = [x for x in file_list if os.listdir(x)]
    train_ls = []
    test_ls = []
    if "VIPL" == name:
        subjects = [x.split("/")[-1].split("_")[0] for x in file_list]
        # 该函数是去除数组中的重复数字，并进行排序之后输出
        uni_subject = np.unique(subjects) 
        for train_idx, test_idx in kf.split(uni_subject):
            train_subject = [uni_subject[x] for x in train_idx]
            test_subject = [uni_subject[x] for x in test_idx]
            train_video = [x for x in file_list if x.split("/")[-1].split("_")[0] in train_subject]
            test_video = [x for x in file_list if x.split("/")[-1].split("_")[0] in test_subject]
            train_ls.append(train_video)
            test_ls.append(test_video)

#         fold_data_dict = {}
#         fold_files = glob.glob("/liqi/NiuXueSongforHR/VIT_MSTmap_UBFC/src_code/Utils/fold_split/*.mat")
#         for fold in fold_files:
#             name = fold.split('/')[-1].split('.')[0]
#             fold_data = scipy.io.loadmat(fold)
#             fold_data_dict[name] = fold_data[name]
#         subjects = [x+1 for x in range(107)]
#         for idx, fold in enumerate(fold_data_dict.keys()):
#             test_subject = [f'p{x}_' for x in fold_data_dict[fold].squeeze(0)]
#             # test_subject = list(map(str, test_subject))
#             train_subject = [f'p{x}_' for x in subjects if f'p{x}_' not in test_subject]
#             # train_subject = list(map(str, train_subject))
#             train_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in train_subject]
#             test_video = [x for x in file_list if f'{x.split("/")[-1].split("_")[0]}_' in test_subject]
#             train_ls.append(train_video)
#             test_ls.append(test_video)

    elif "HCI" == name:
        # E:\MSTforHCI\subject_6_666
        subjects = [x.split("/")[-1].split("_")[1] for x in file_list]
        uni_subject = np.unique(subjects)
        for train_idx, test_idx in kf.split(uni_subject):
            train_subject = uni_subject[train_idx]
            test_subject = uni_subject[test_idx]
            train_video = [x for x in file_list if x.split("/")[-1].split("_")[1] in train_subject]
            test_video = [x for x in file_list if x.split("/")[-1].split("_")[1] in test_subject]
            train_ls.append(train_video)
            test_ls.append(test_video)

    else:
        print("not find such dataset!")

    return train_ls, test_ls


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(20)
    map_path = config.MST_MAP_PATH
    maps_path_per_video = glob.glob(os.path.join(map_path, '*'))
#     count = 0
#     for x in maps_path_per_video:
#         if not os.listdir(x):
#             count += 1
#             print(x)
    
#     print(f"done! {count}")

    print(len(maps_path_per_video))
    print(maps_path_per_video[10])
    traindata, testdata = SplitforSubject(maps_path_per_video, "VIPL")
    print(len(traindata[0]))
    print(len(testdata[0]))
    for train, test in zip(traindata, testdata):
        sub1 = np.unique([x.split("/")[-1].split("_")[0] for x in train])
        sub2 = np.unique([x.split("/")[-1].split("_")[0] for x in test])
        print(sub1)
        print(sub2)
        same = [x for x in sub1 if x in sub2]
        print(same)
        break