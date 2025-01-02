######################################
#
# 主要使用calculate_metric_peak_per_video
# 和calculate_metric_per_video两个函数，前者
# 是峰值检测，对短信号可以设窗口长度等于信号长度，
# 后者是功率谱，不用设长度，两个函数输入预测值和
# 真值，输出相应的信号，HR是预测值的，HR0是真值的
#
#####################################

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

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

def mag2db(mag):

    return 20. * np.log10(mag)

# 计算心率值，通过已经计算出来的信号的频率域，取最大值对应的频率，再将该频率乘以60得到每分钟的心率
# def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
def calculate_HR(pxx_pred, frange_pred, fmask_pred):
    pred_HR = np.take(frange_pred, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    # ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR

# 计算信号的信噪比，
def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR/60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp


def calculate_metric_per_video(predictions, signal='pulse', fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.6/ fs * 2, 3 / fs * 2], btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

#     if signal == 'pulse':
#         pred_window = detrend(np.cumsum(predictions), 100)
#         # label_window = detrend(np.cumsum(labels), 100)
#     else:
#         pred_window = np.cumsum(predictions)

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(predictions))
        # label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

    pred_window = np.expand_dims(pred_window, 0)
    # label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    # 计算功率谱，在将得到的频率域计算心率
    # N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=300, detrend=False)
    if signal == 'pulse':
        fmask_pred = np.argwhere((f_prd >= 0.6) & (f_prd <= 3))  # regular Heart beat are 0.75*60 and 2.5*60
    else:
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
    pred_window = np.take(f_prd, fmask_pred)
    # Labels FFT
    # f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=N, detrend=False)
    # if signal == 'pulse':
    #     fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
    # else:
    #     fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
    # label_window = np.take(f_label, fmask_label)

    # MAE
    # temp_HR, temp_HR_0 = calculate_HR(pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
    temp_HR = calculate_HR(pxx_pred, pred_window, fmask_pred)
    # temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)

    return temp_HR


if __name__ == "__main__":
    pass
#     pred = np.load(r"G:\\WeChat\\WeChat Files\\wxid_bg0zwvjac6fg22\\FileStorage\\File\\2021-12\\pred.npy", allow_pickle=True)[10]
#     label = np.load(r"G:\\WeChat\\WeChat Files\\wxid_bg0zwvjac6fg22\\FileStorage\\File\\2021-12\\label.npy", allow_pickle=True)[10]
#     print(f"pred len: {len(pred)}; label len:{len(label)}")
    
#     plt.subplot(2,1,1)
#     plt.plot(pred)
#     plt.title('pred bvp')
#     plt.subplot(2,1,2)
#     plt.plot(label)
#     plt.title('label bvp')
#     plt.show()
#     hr, hr0 = calculate_metric_per_video(pred, label)
#     print(f"hr:{hr}, hr0:{hr0}")