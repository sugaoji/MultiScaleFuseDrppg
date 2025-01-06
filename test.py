import scipy.io
import matplotlib.pyplot as plt
import torch
# Load the .mat file
mat_file_path = './img/bvp.mat'
data = scipy.io.loadmat(mat_file_path)

# Access the data
bvp_data = data.get('bvp')

def draw_bvp(bvp, save_path):
    plt.figure(figsize=(10, 3))
    plt.plot(bvp[0])
    plt.title('BVP Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(save_path)

def draw_two_bvp(bvp1, bvp2, save_path):
    plt.figure(figsize=(10, 3))
    plt.plot(bvp1[0], color='red')
    plt.plot(bvp2[0], color='blue')
    plt.title('BVP Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(save_path)
# Example usage of the function
draw_bvp(bvp_data, './img/bvp_origin.png')
tensor_bvp = torch.tensor(bvp_data)
fft_bvp = torch.fft.fft(tensor_bvp, norm='ortho', dim=1)
ifft_bvp = torch.fft.ifft(fft_bvp[:,:150], norm='ortho', dim=1).real

# Linear interpolation to resize ifft_bvp back to 300
ifft_bvp_resized = torch.nn.functional.interpolate(ifft_bvp.unsqueeze(0), size = 300, mode='linear', align_corners=True)

ifft_bvp = ifft_bvp_resized.squeeze(0)

draw_two_bvp(ifft_bvp, bvp_data, './img/bvp_ifft.png')