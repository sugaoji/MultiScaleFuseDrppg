import math
import torch
import os
import torch.nn.functional as F

from einops import rearrange
from tqdm import tqdm
import config
import matplotlib
import torch_dct

from Models import diff_model

matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.switch_backend( 'agg')
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import sys
from Utils import post_process




def train_fn(model, data_loader, optimizer, lossfunc_pearson, lossfunc_mse):
    model.train()
    loss_per_batch = []
    target_bvp_per_batch = []
    target_hr_per_batch = []
    predicted_bvp_per_batch = []
    
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    for data in tk_iterator:
        maps = data['st_maps'].cuda()
        gt_HR = data['gt_HR'].cuda()
        bvp = data['bvp'].cuda()
        optimizer.zero_grad()

        bvp_75 = torch.fft.ifft(torch.fft.fft(bvp, norm='ortho',dim=1)[:, :75], norm='ortho',dim=1).real
        bvp_150 = torch.fft.ifft(torch.fft.fft(bvp, norm='ortho',dim=1)[:, :150], norm='ortho',dim=1).real 
        ecg = model(maps,bvp)
        ecg = (ecg-torch.mean(ecg)) /torch.std(ecg)

        ecg_75 = torch.fft.ifft(torch.fft.fft(ecg, norm='ortho',dim=1)[:, :75], norm='ortho',dim=1).real
        ecg_150 = torch.fft.ifft(torch.fft.fft(ecg, norm='ortho',dim=1)[:, :150], norm='ortho',dim=1).real
        loss_ecg = lossfunc_pearson(ecg, bvp)
        loss_ecg_75 = lossfunc_pearson(ecg_75, bvp_75)
        loss_ecg_150 = lossfunc_pearson(ecg_150, bvp_150)
        loss_total = loss_ecg + (loss_ecg_75 + loss_ecg_150) * 0.1

        loss_total.backward()
        optimizer.step()
        
        target_hr_per_batch.extend(gt_HR.detach().cpu().numpy())
        target_bvp_per_batch.extend(bvp.detach().cpu().numpy())
        predicted_bvp_per_batch.extend(ecg.detach().cpu().numpy())
        loss_per_batch.append(loss_total.item())
        
    return target_hr_per_batch, target_bvp_per_batch, predicted_bvp_per_batch, loss_per_batch


def eval_fn(model_eval_temp, data_loader, lossfunc_pearson, lossfunc_mse):
    model_eval_temp.eval()
    loss_per_batch = []
    target_bvp_per_batch = []
    target_hr_per_batch = []
    predicted_bvp_per_batch = []
    predicted_hr_per_batch = []
    path_per_batch = []

    with torch.no_grad():
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            maps = data['st_maps'].cuda()
            gt_HR = data['gt_HR'].cuda()
            bvp = data['bvp'].cuda()
            path = data['path']

            bvp_75 = torch.fft.ifft(torch.fft.fft(bvp, norm='ortho',dim=1)[:, :75], norm='ortho',dim=1).real
            bvp_150 = torch.fft.ifft(torch.fft.fft(bvp, norm='ortho',dim=1)[:, :150], norm='ortho',dim=1).real 
            
            ecg = model_eval_temp(maps,bvp)
            ecg = (ecg-torch.mean(ecg)) /torch.std(ecg)

            ecg_75 = torch.fft.ifft(torch.fft.fft(ecg, norm='ortho',dim=1)[:, :75], norm='ortho',dim=1).real
            ecg_150 = torch.fft.ifft(torch.fft.fft(ecg, norm='ortho',dim=1)[:, :150], norm='ortho',dim=1).real
            
            loss_ecg = lossfunc_pearson(ecg, bvp)
            
            loss_ecg_75 = lossfunc_pearson(ecg_75, bvp_75)
            loss_ecg_150 = lossfunc_pearson(ecg_150, bvp_150)
            loss_total = loss_ecg + (loss_ecg_75 + loss_ecg_150) * 0.1


            folder_path = config.pred_folder_path
            if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(folder_path)

            with parallel_backend("loky"):
                Parallel(n_jobs=-1)(delayed(process_rppg)(ecg.detach().cpu().numpy(), subject,folder_path)
                                   for subject in range(ecg.shape[0]))
            for subject in range(ecg.shape[0]):
                file_path = os.path.join(folder_path,f"{subject}.npy")
                predicted_hr_per_batch.append(np.load(file_path).item())
                target_hr_per_batch.append(gt_HR.squeeze(1)[subject].item())

            target_bvp_per_batch.extend(bvp.detach().cpu().numpy())
            predicted_bvp_per_batch.extend(ecg.detach().cpu().numpy())
            loss_per_batch.append(loss_total.item())
            path_per_batch.extend(path)

    return target_hr_per_batch, predicted_hr_per_batch, target_bvp_per_batch, predicted_bvp_per_batch, loss_per_batch, path_per_batch


def process_rppg(predicted_per_batch, subject,folder_path):
    predict_hr = post_process.calculate_metric_per_video(predicted_per_batch[subject])
    np.save(os.path.join(folder_path,f"{subject}.npy"), predict_hr)





