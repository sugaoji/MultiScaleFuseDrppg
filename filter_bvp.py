import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
from scipy.io import loadmat
import scipy
from scipy import misc, io
from scipy.signal import butter
from scipy.sparse import spdiags

import config


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

# 该函数是去除信号的趋势，比如上升趋势，或者下降趋势
def detrend(signal, Lambda):
    signal_length = signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

# 计算心率值，通过已经计算出来的信号的频率域，取最大值对应的频率，再将该频率乘以60得到每分钟的心率
def calculate_HR(pxx_label, frange_label, fmask_label):
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return ground_truth_HR

def calculate_metric_per_video(labels, signal='pulse', fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.66 / fs * 2, 3.0 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

#     if signal == 'pulse':
#         label_window = detrend(np.cumsum(labels), 100)

    if bpFlag:
        label_window = scipy.signal.filtfilt(b, a, np.double(labels))
    
    return label_window

#     label_window = np.expand_dims(label_window, 0)
#     # 计算功率谱，在将得到的频率域计算心率
# #     N = next_power_of_2(label_window.shape[1])
#     # Labels FFT
#     f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=300, detrend=False)
#     if signal == 'pulse':
#         fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
#     else:
#         fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
#     label_window = np.take(f_label, fmask_label)
#     temp_HR = calculate_HR(pxx_label, label_window, fmask_label)
#     return temp_HR


def gen(bvp_path):
    # VIPL和PURE数据集squeeze(0)，UBFC数据集squeeze(1)
    sdf= loadmat(bvp_path)['bvp']
    bvp = loadmat(bvp_path)['bvp'].squeeze(0)
    hr_temp = calculate_metric_per_video(bvp)
    HR = {'bvp':hr_temp}
    hr_path = os.path.join(f'{config.filter_path}', bvp_path.split('/')[-3], bvp_path.split('/')[-2])
    io.savemat(hr_path + f"/filter_bvp.mat", HR)


if __name__ == "__main__":
    bvp_path_list = glob.glob(f'{config.filter_path}/*/*/bvp.mat')
    with parallel_backend("loky"):
        Parallel(n_jobs=-1)(delayed(gen)(bvp_path) for bvp_path in tqdm(bvp_path_list))


    # print(len(bvp_path_list))
    # gen('/qianwei/dataset/preUBFC/MSTmap_dlib/subject1/1/bvp.mat')
    # target = loadmat('/qianwei/dataset/preUBFC/MSTmap_dlib/subject1/1/gt_HR.mat')
    # print(target)

    # for i in range(12):
    #     data = pd.read_excel(f'/qianwei/dataset/preUBFC/UBFC_wave/subject{38+i}.xlsx', header=None)
    #     hr = calculate_metric_peak_per_video_clip(np.array(data).squeeze(1))
    #     print(hr)
