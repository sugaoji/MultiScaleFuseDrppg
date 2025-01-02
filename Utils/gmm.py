import torch

# def gmm_noise(shape=[3,300]):
#     # 定义高斯混合模型的参数
#     num_components = 3  # 高斯分布的数量
#     means = torch.tensor([0.0, 4.0, -4.0])  # 各个高斯分布的均值
#     stds = torch.tensor([1.0, 0.5, 0.5])  # 各个高斯分布的标准差
#     weights = torch.tensor([0.5, 0.25, 0.25])  # 各个高斯分布的混合权重（必须归一化）
#
#     noise = torch.zeros(size=shape,device='cuda')
#     # 确保权重之和为 1
#     weights = weights / torch.sum(weights)
#
#     for index in range(shape[0]):
#
#         # 定义采样数量为 300
#         num_samples = 300
#
#         # 1. 从离散分布中选择一个高斯分布的索引
#         components = torch.multinomial(weights, num_samples, replacement=True)
#
#         # 2. 从选择的高斯分布中采样
#         samples = torch.empty(num_samples)
#         for i in range(num_components):
#             mask = components == i
#             num_samples_i = mask.sum().item()
#             samples[mask] = torch.normal(means[i], stds[i], size=(num_samples_i,))
#
#         # 输出生成的 300 个样本
#         noise[index] = samples
#     return noise

def gmm_noise(shape=[3, 300]):
    return torch.randn(size=shape,device='cuda')
