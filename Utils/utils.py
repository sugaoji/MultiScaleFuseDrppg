import numpy as np

def collate_fn(batch):
    """
    :param batch:
    :return: list 一个batch size大小的数据
    """
    return batch

def rmse(l1, l2):
    return np.sqrt(np.mean((l1-l2)**2))

def mae(l1, l2):
    return np.mean([abs(item1-item2)for item1, item2 in zip(l1, l2)])

def r(l1, l2):
    return np.sum((l1-np.mean(l1))*(l2-np.mean(l2)))/(np.sqrt(np.sum((l1-np.mean(l1))**2))*np.sqrt(np.sum((l2-np.mean(l2))**2)))

def mer(l1,l2):
    return 100*np.sum(abs(l1-l2)/l1)/l1.shape[0]

def std(l1, l2):
    mean_err = np.mean(l1-l2)
    return np.sqrt(np.mean((abs(l1-l2)-mean_err)**2))

def compute_criteria(target_hr_array, predicted_hr_array):
    """
    :param target_hr_list: 一维数据
    :param predicted_hr_list: 一维数据
    :return: 字典，包含MAE 和RMSE
    """
    pearson_per_signal = []
    # HR_MAE = mae(np.array(predicted_hr_list), np.array(target_hr_list))
    # HR_RMSE = rmse(np.array(predicted_hr_list), np.array(target_hr_list))

    HR_MAE = mae(target_hr_array, predicted_hr_array)
    HR_RMSE = rmse(target_hr_array, predicted_hr_array)
    HR_MER = mer(target_hr_array, predicted_hr_array)
    HR_r = r(target_hr_array, predicted_hr_array)
    HR_std = std(target_hr_array, predicted_hr_array)

    # for (gt_signal, predicted_signal) in zip(target_hr_list, predicted_hr_list):
    #     r, p_value = pearsonr(predicted_signal, gt_signal)
    #     pearson_per_signal.append(r)

    # return {"MAE": np.mean(HR_MAE), "RMSE": HR_RMSE, "Pearson": np.mean(pearson_per_signal)}
    return {"MAE": HR_MAE, "RMSE": HR_RMSE, 'MER':HR_MER, 'r':HR_r, 'std':HR_std}