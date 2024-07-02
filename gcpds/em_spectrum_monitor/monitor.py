from pyhackrf2 import HackRF
import numpy as np

########################################################################
class Scanning:
    """"""

    # ----------------------------------------------------------------------
    def __init__(self):
        """Constructor"""
        self.hackrf = HackRF()
        self.hackrf.filter_bandwidth = 20e6
        self.hackrf.sample_count_limit = 5e5
        self.hackrf.vga_gain = 16
        self.hackrf.lna_gain = 0
        self.hackrf.amplifier_on = False
        self.hackrf.sample_rate = 20e6

        self.overlap = 0
        self.samples_to_read = 1e3


    # ----------------------------------------------------------------------
    def scan(self, start, end):
        """"""
        step = self.hackrf.sample_rate
        wide_samples = []
        for center_freq in range(int(start), int(end), int(step - self.overlap)):
            self.hackrf.center_freq = center_freq
            samples = np.array(self.hackrf.read_samples(self.samples_to_read))
            wide_samples.append({
                'start': center_freq,
                'end': center_freq + step - self.overlap,
                'samples': samples,
            })
        return wide_samples



    # ----------------------------------------------------------------------
    def read(self):
        """"""


    # ----------------------------------------------------------------------
    def write(self):
        """"""
