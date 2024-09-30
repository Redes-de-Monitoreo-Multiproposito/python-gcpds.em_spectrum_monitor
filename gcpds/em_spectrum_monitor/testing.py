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

# samples = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')
# Pxx, f = plt.psd(samples, 4098, 20)
# plt.close()

# plt.plot(f, 10 * np.log10(Pxx))
# plt.show()

# hackrf = HackRF()
# hackrf.vga_gain = 0
# hackrf.lna_gain = 0
# hackrf.center_freq = 98e6
# hackrf.sample_rate = 20e6
# samples = hackrf.read_samples(200000)
# # samples = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

# pros = Processing()
# f, Pxx = pros.welch(samples,20e6)
# f = np.linspace(88, 108, len(Pxx))

# # Percentil inferior (por ejemplo, 10%) para estimar el ruido
# noise_level = np.percentile(Pxx, 80)

# # Percentil superior (por ejemplo, 90%) para estimar la señal
# signal_level = np.percentile(Pxx, 90)

# # Umbral como punto medio entre ruido y señal
# threshold = (noise_level + signal_level) / 1.5 + 0.4 * ((noise_level + signal_level) / 1.5)

# # # plt.stairs(counts, bins)
# # # plt.axvline(threshold)
# # # plt.show()

# plt.plot(f, 10 * np.log10(Pxx))
# # plt.ylim(1e-8 ,2.5e-5)
# # plt.axhline(noise_level)
# # plt.axhline(threshold)
# plt.show()

# plt.semilogy(f, Pxx)
# plt.show()

# rds = RDS()
# df_12h = rds.parameter(hours_to_scan=4)
# print(df_12h.head(20))

# print(f'promedio 5m: {parameters_prom_12h[0]}')
# print(f'promedio de promedios 12h: {parameters_prom_12h_final}')

scan =  Scanning(time_to_read=0.01)
wide_samples1 = scan.scan(88e6, 108e6)
samples2 = scan.concatenate(wide_samples1, 'mean')
samples2 = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

pros = Processing()
# f1, Pxx1 = pros.welch(samples1, 20e6)   #Sin señal
f, Pxx2 = pros.welch(samples2, 20e6)   #Con señal
# f = np.linspace(88, 108, len(Pxx2))
# plt.semilogy(f, Pxx2)
# # plt.show()

detect = Detection()
# detect.antenna_detection()

# peak_freqs1, peak_powers1, _, thres1 = detect.power_based_detection(f, Pxx1)
peak_powers, peak_freqs, threshold, noise_lvl = detect.power_based_detection(f, Pxx2)

# plt.semilogy(f, Pxx1, c='red')
# plt.axhline(thres1, c='red')
# plt.scatter(peak_freqs1, peak_powers1, marker='x', c='red')

# plt.semilogy(f, Pxx2, c='blue')
# plt.axhline(thres2, c='blue')
# plt.axhline(noise_lvl2, c='red')
# plt.scatter(peak_freqs2, peak_powers2, marker='x', c='red')

# plt.ylim(2.5e-13, 1e-9)
# plt.title('Power Spectral Density')
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()