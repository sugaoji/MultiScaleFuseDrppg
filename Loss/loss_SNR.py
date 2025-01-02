import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class TorchLossComputer(object):

    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator

    @staticmethod 
    def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs):

        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        
        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        y = target.view((1)).type(torch.long)

        CE = F.cross_entropy(complex_absolute, y)

        
        return CE