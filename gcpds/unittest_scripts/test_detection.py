import unittest
import numpy as np
from gcpds.em_spectrum_monitor.processing import Processing
from gcpds.em_spectrum_monitor.detection import Detection

class TestScanning(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(self):
        """
        Initial setup for the detection object
        """
        signal = np.load('database/Samples 88.0 and 108.0MHz with time to read 0.01s and 0MHz overlap.npy')
        pros = Processing()
        f, Pxx = pros.welch(signal, 20e6)

        self.detec = Detection()
        self.peak_freqs, self.peak_powers, self.detections = self.detec.power_based_detection(f, Pxx)

    def test_detection(self):
        self.assertIsInstance(self.peak_freqs, np.ndarray)
        self.assertIsInstance(self.peak_powers, np.ndarray)
        self.assertIsInstance(self.detections, list)
    

if __name__ == '__main__':
    unittest.main()
