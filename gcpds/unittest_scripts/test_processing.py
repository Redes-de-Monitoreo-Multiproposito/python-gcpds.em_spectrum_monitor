import unittest
import numpy as np
from gcpds.em_spectrum_monitor.monitor import Scanning
from gcpds.em_spectrum_monitor.processing import Processing

class TestScanning(unittest.TestCase):
    """"""

    @classmethod
    def setUpClass(cls, method: str = 'inter'):
        """
        Initialize the Scanning object with the given parameters for unit testing.
        """
        cls.method = method
        cls.scanning = Scanning(vga_gain=0, lna_gain=0, sample_rate=20e6, overlap=0, time_to_read=0.01)
        cls.wide_samples = cls.scanning.scan(88e6, 108e6)
        cls.iq_samples = cls.scanning.concatenate(cls.wide_samples, 'mean')

        cls.processing = Processing()
        cls.real_samples = cls.processing.convert_to_real(cls.iq_samples, method)
        cls.fft_real_samples = cls.processing.fft(cls.real_samples)
        cls.fft_iq_samples = cls.processing.fft(cls.iq_samples)
        cls.f_real, cls.Pxx_real = cls.processing.welch(cls.real_samples)
        cls.f_iq, cls.Pxx_iq = cls.processing.welch(cls.iq_samples)

    def test_convert_to_real(self):
        """
        Check the correct shape of real signal.
        """
        if self.method == 'inter':
            self.assertEqual(len(self.real_samples), 2*len(self.iq_samples))
        elif self.method == 'sincos':
            self.assertEqual(len(self.real_samples), len(self.iq_samples))

    def test_fft(self):
        """
        Check the correct shape of fourier transform
        """
        self.assertEqual(len(self.fft_real_samples), len(self.iq_samples))
        self.assertEqual(len(self.fft_iq_samples), len(self.iq_samples)//2)

    def test_welch(self):
        """
        Check the correct shape of power spectral density using welch method.
        """
        self.assertTrue(len(self.Pxx_real) == len(self.f_real) == 513)
        self.assertTrue(len(self.Pxx_iq) == len(self.f_iq) == 1024)



if __name__ == '__main__':
    unittest.main()