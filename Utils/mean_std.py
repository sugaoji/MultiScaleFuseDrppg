import torch
import numpy as np
import glob
import os
import torchvision.transforms as T
from tqdm import tqdm
import sys
os.chdir(sys.path[0])
sys.path.append("..")
import config
from Dataset_loaders.dataset import DataLoader


if __name__ == "__main__":
    
    Map_path = config.MST_MAP_PATH
    Maps_path_per_video = glob.glob(os.path.join(Map_path, '*'))
    
    test = [f'{Map_path}/subject{x}' for x in range(38,50)]
    train = [x for x in Maps_path_per_video if x not in test]

    # MSTmap for UBFC success1
    transform_train = T.Compose([T.ToTensor()])
    transform_test = T.Compose([T.ToTensor()])

    dataset_train = DataLoader(Maps_path_per_video, transforms=transform_train)
    # dataset_train = DataLoader(train, transforms=transform_train)


    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=128,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    # compute the mean and std of map
    tk_iterator = tqdm(train_loader, total=len(train_loader))
    num_of_pixels_R = len(dataset_train)*300*63
    num_of_pixels_G = len(dataset_train)*300*63
    num_of_pixels_B = len(dataset_train)*300*63
    num_of_pixels_Y = len(dataset_train)*300*63
    num_of_pixels_U = len(dataset_train)*300*63
    num_of_pixels_V = len(dataset_train)*300*63
    total_sum_R = 0
    total_sum_G = 0
    total_sum_B = 0
    total_sum_Y = 0
    total_sum_U = 0
    total_sum_V = 0
    for batch in tk_iterator:
        total_sum_R += batch['st_maps'].cuda()[:,0,:,:].sum()
        total_sum_G += batch['st_maps'].cuda()[:,1,:,:].sum()
        total_sum_B += batch['st_maps'].cuda()[:,2,:,:].sum()
        total_sum_Y += batch['st_maps'].cuda()[:,3,:,:].sum()
        total_sum_U += batch['st_maps'].cuda()[:,4,:,:].sum()
        total_sum_V += batch['st_maps'].cuda()[:,5,:,:].sum()
    mean_R = total_sum_R / num_of_pixels_R
    mean_G = total_sum_G / num_of_pixels_G
    mean_B = total_sum_B / num_of_pixels_B
    mean_Y = total_sum_Y / num_of_pixels_Y
    mean_U = total_sum_U / num_of_pixels_U
    mean_V = total_sum_V / num_of_pixels_V

    print(f'mean_R:{mean_R} mean_G:{mean_G} mean_B:{mean_B} mean_Y:{mean_Y} mean_U:{mean_U} mean_V:{mean_V}')
    
    sum_of_squared_error_R = 0
    sum_of_squared_error_G = 0
    sum_of_squared_error_B = 0
    sum_of_squared_error_Y = 0
    sum_of_squared_error_U = 0
    sum_of_squared_error_V = 0
    for batch in tk_iterator:
        sum_of_squared_error_R += ((batch['st_maps'].cuda()[:,0,:,:] - mean_R).pow(2)).sum()
        sum_of_squared_error_G += ((batch['st_maps'].cuda()[:,1,:,:] - mean_G).pow(2)).sum()
        sum_of_squared_error_B += ((batch['st_maps'].cuda()[:,2,:,:] - mean_B).pow(2)).sum()
        sum_of_squared_error_Y += ((batch['st_maps'].cuda()[:,3,:,:] - mean_Y).pow(2)).sum()
        sum_of_squared_error_U += ((batch['st_maps'].cuda()[:,4,:,:] - mean_U).pow(2)).sum()
        sum_of_squared_error_V += ((batch['st_maps'].cuda()[:,5,:,:] - mean_V).pow(2)).sum()
    std_R = torch.sqrt(sum_of_squared_error_R / num_of_pixels_R)
    std_G = torch.sqrt(sum_of_squared_error_G / num_of_pixels_G)
    std_B = torch.sqrt(sum_of_squared_error_B / num_of_pixels_B)
    std_Y = torch.sqrt(sum_of_squared_error_Y / num_of_pixels_Y)
    std_U = torch.sqrt(sum_of_squared_error_U / num_of_pixels_U)
    std_V = torch.sqrt(sum_of_squared_error_V / num_of_pixels_V)
    
    print(f'std_R:{std_R} std_G:{std_G} std_B:{std_B} std_Y:{std_Y} std_U:{std_U} std_V:{std_V}')