import matplotlib.pyplot as plt
import numpy as np
from monitor import Scanning
from processing import Processing
from detection import Detection
from scipy.signal import find_peaks

scan =  Scanning(vga_gain=0, time_to_read=0.01)
wide_samples1 = scan.scan(88e6, 108e6)
samples1 = scan.concatenate(wide_samples1, 'mean')
samples2 = np.load('database/Samples 88.0 and 108.0MHz with time to read 0.01s and 0MHz overlap.npy')
#wide_samples2 = scan.scan(88e6, 108e6)
#samples2 = scan.concatenate(wide_samples2, 'mean')

pros = Processing()
f1, Pxx1 = pros.welch(samples1, 20e6)   #Sin señal
f2, Pxx2 = pros.welch(samples2, 20e6)   #Con señal
f = np.linspace(88, 108, len(Pxx1))

# hist1, bin1 = np.histogram(Pxx1, bins=500)    # Sin antena
# hist2, bin2 = np.histogram(Pxx2, bins=500)    # Con antena

# plt.stairs(hist1, bin1)
# plt.title('Pxx1')
# plt.show()
# plt.stairs(hist2, bin2)
# plt.title('Pxx2')
# plt.show()

# def threshold(hist, bin):
#     max_rep_indice = np.argmax(hist)

#     value = bin[max_rep_indice+1]

#     return 1.5 * value

# thres1 = threshold(hist1, bin1) #Sin señal 
# thres2 = threshold(hist2, bin2) #Con señal

detect = Detection()


peak_freqs1, peak_powers1, _, thres1 = detect.power_based_detection(f, Pxx1)
peak_freqs2, peak_powers2, _, thres2 = detect.power_based_detection(f, Pxx2)

plt.semilogy(f, Pxx1, c='red')
plt.axhline(thres1, c='red')
plt.scatter(peak_freqs1, peak_powers1, marker='x', c='red')

plt.semilogy(f, Pxx2, c='blue')
plt.axhline(thres2, c='blue')
plt.scatter(peak_freqs2, peak_powers2, marker='x', c='red')

plt.ylim(2.5e-13, 1e-9)
plt.title('Power Spectral Density')
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# plt.boxplot(Pxx2)
# plt.show()