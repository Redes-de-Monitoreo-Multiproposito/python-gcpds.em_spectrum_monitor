import matplotlib.pyplot as plt
import numpy as np
from monitor import Scanning
from processing import Processing
from detection import Detection
from RDS import RDS
from scipy.signal import find_peaks


rds = RDS()
rds.broadcasters('Manizales')
parameters_prom_12h_final, parameters_prom_12h = rds.parameter(hours_to_scan=12)

# print(f'promedio 5m: {parameters_prom_12h[0]}')
# print(f'promedio de promedios 12h: {parameters_prom_12h_final}')

# scan =  Scanning(vga_gain=0, time_to_read=0.01)
# wide_samples1 = scan.scan(88e6, 108e6)
# # samples1 = scan.concatenate(wide_samples1, 'mean')
# samples2 = np.load('database/Samples 88.0 and 108.0MHz with time to read 0.01s and 0MHz overlap.npy')

# pros = Processing()
# f1, Pxx1 = pros.welch(samples1, 20e6)   #Sin señal
# f, Pxx2 = pros.welch(samples2, 20e6)   #Con señal
# f = np.linspace(88, 108, len(Pxx2))
# plt.semilogy(f, Pxx2)
# plt.show()

# detect = Detection()
# detect.antenna_detection()

# peak_freqs1, peak_powers1, _, thres1 = detect.power_based_detection(f, Pxx1)
# peak_freqs2, peak_powers2, _, thres2 = detect.power_based_detection(f, Pxx2)

# plt.semilogy(f, Pxx1, c='red')
# plt.axhline(thres1, c='red')
# plt.scatter(peak_freqs1, peak_powers1, marker='x', c='red')

# plt.semilogy(f, Pxx2, c='blue')
# plt.axhline(thres2, c='blue')
# plt.scatter(peak_freqs2, peak_powers2, marker='x', c='red')

# plt.ylim(2.5e-13, 1e-9)
# plt.title('Power Spectral Density')
# plt.legend()
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()