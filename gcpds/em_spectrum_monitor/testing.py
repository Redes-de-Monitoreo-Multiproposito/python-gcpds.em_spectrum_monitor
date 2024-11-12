import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from monitor import Scanning
from processing import Processing
from detection import Detection
from RDS import RDS
from scipy.signal import find_peaks, peak_widths
from hackrf import HackRF
from gcpds.filters import frequency as flt
from matplotlib import rcParams

pros = Processing()
hackrf = HackRF()
center_freq = 470e6
sample_rate = 20e6
vga_gain = 0
lna_gain = 0
num_samples = 20000000

hackrf.vga_gain = vga_gain
hackrf.lna_gain = lna_gain
hackrf.sample_rate = sample_rate
hackrf.center_freq = center_freq

samples = hackrf.read_samples(num_samples)
np.save(f'5G(center freq: {center_freq}, sample rate: {sample_rate}, vga gain: {vga_gain}, lna gain: {lna_gain},num samples: {num_samples}(1).npy)', samples)

f, Pxx = signal.welch(samples, fs=20, nperseg=32768/4)

f = np.linspace((center_freq/1e6)-10, (center_freq/1e6)+10, len(Pxx))
Pxx = np.fft.fftshift(Pxx)

plt.plot(f, 10*np.log10(Pxx))
plt.show()
# samples = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

# f, Pxx = pros.welch(samples, 20)
# f = np.linspace(center_freq/1e6 - sample_rate/1e6/2, center_freq/1e6+sample_rate/1e6/2, len(Pxx))

# # index = np.where(np.isclose(f, center_freq/1e6, atol=0.01))[0][0]
# # print(np.sqrt(Pxx[index]*377))

# plt.plot(f, 10*np.log10(Pxx))
# # plt.ylim(-62, -36)
# plt.show()

# rds = RDS()
# df_12h = rds.parameter(hours_to_scan=4)

# print(f'promedio 5m: {parameters_prom_12h[0]}')
# print(f'promedio de promedios 12h: {parameters_prom_12h_final}')

# scan =  Scanning(time_to_read=0.01)
# wide_samples1 = scan.scan(88e6, 108e6)
# samples2 = scan.concatenate(wide_samples1, 'mean')
# samples2 = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

# pros = Processing()
# # # f1, Pxx1 = pros.welch(samples1, 20e6)   #Sin señal
# # f, Pxx2 = pros.welch(samples2, 20e6)   #Con señal
# # f = np.linspace(88, 108, len(Pxx2))
# # plt.semilogy(f, Pxx2)
# # # plt.show()

# samples = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

# f, Pxx = pros.welch(samples, 20)

# plt.psd(samples, 1024, 20)
# plt.plot(f, 10*np.log10(Pxx), '--')
# plt.show()

