import numpy as np
import time
from hackrf import HackRF
import matplotlib.pyplot as plt
from detection import Detection
from processing import Processing

class VHF1():
    def __init__(self):
        self.detec = Detection()
        self.pros = Processing()
        self.frequencies = self.detec.broadcasters('Manizales')

    def parameter(self):
        sample = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

        parameters = {
            'time': [],
            'freq': [],
            'power': [],
            'snr': []
        }

        t0 = time.strftime('%X')

        f, Pxx = self.pros.welch(sample)
        f = (f + 98e6) / 1e6

        _, peak_freqs, _ = self.detec.power_based_detection(f, Pxx)
        
        filtered_peak_freqs = []

        for peak_freq in peak_freqs:
            is_redundant = any(np.isclose(peak_freq, center_freq, atol=0.01) for center_freq in self.frequencies)
            if not is_redundant:
                filtered_peak_freqs.append(peak_freq)

        center_freqs = np.sort(np.concatenate((self.frequencies, filtered_peak_freqs)))

        for center_freq in center_freqs:
            index = np.where(np.isclose(f, center_freq, atol=0.01))[0][0]

            lower_index = np.argmin(np.abs(f - (f[index] - 0.125)))
            upper_index = np.argmin(np.abs(f - (f[index] + 0.125)))

            freq_range = f[lower_index:upper_index + 1]
            Pxx_range = Pxx[lower_index:upper_index + 1]      

            power = np.trapz(Pxx_range, freq_range)

            parameters['time'].append(t0)
            parameters['freq'].append(center_freq)
            parameters['power'].append(10 * np.log10(power))
            parameters['snr'].append(10 * np.log10(power / Pxx[0]))
            
        print(parameters)
       
vhf1 = VHF1()
vhf1.parameter()