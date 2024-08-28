import numpy as np
import matplotlib.pyplot as plt
import utils.frft as frft
from tqdm import tqdm
import torch
from numpy.fft import fftshift
# from scipy.signal import chirp, frft
'''
def plot_frft(iq_data, alpha):
    # 计算分数阶傅立叶变换
    frft_result = frft(iq_data, alpha)

    # 绘制时域图
    plt.subplot(2, 1, 1)
    plt.plot(iq_data)
    plt.title('Time Domain')

    # 绘制分数阶频域图
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(frft_result))
    plt.title(f'Fractional Fourier Transform (alpha={alpha})')

    plt.tight_layout()
    plt.show()
'''

'''
def plot_time_freq(iq_data, sample_rate):
    # 计算FFT
    spectrum = np.fft.fft(iq_data)
    spectrum_magnitude = np.abs(spectrum)

    # 计算频率轴
    freq_axis = np.fft.fftfreq(len(iq_data), d=1/sample_rate)

    # 绘制时域图
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(iq_data)) / sample_rate, iq_data)
    plt.title('Time Domain')

    # 绘制频域图
    plt.subplot(2, 1, 2)
    plt.plot(freq_axis, spectrum_magnitude)
    plt.title('Frequency Domain')

    plt.tight_layout()
    plt.show()
'''

def frft_wrapper(data, alpha=0.8):
    amp, phs = data[0], data[1]
    obj_1d = amp * np.exp( 1.j * phs ) # complex-valued object
    obj_1d_shifted = fftshift( obj_1d )
    obj_1d_shifted_gpu = torch.from_numpy(obj_1d_shifted).cuda()
    fobj_1d = frft.frft(obj_1d_shifted_gpu, alpha)
    return np.absolute(fftshift(fobj_1d.cpu()))

# if __name__ == '__main__':

#     data = np.random.random((2, 1024))
#     # data = fftshift(data)
#     nSnapshots = 11
#     alpha = np.linspace( 0., 2., nSnapshots )
#     for al in tqdm( alpha, total=alpha.size):
#         frft_data = torch.from_numpy(data).cuda()
#         res = frft.frft(frft_data, al)
#         res = fftshift(res)
#         print(res.shape)