import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from monitor import Scanning

class TestScanning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scanning = Scanning(vga_gain=0, lna_gain=0, sample_rate=20e6, overlap=0, time_to_read=0.01)
        cls.wide_samples = cls.scanning.scan(88e6, 108e6)
        cls.samples = cls.scanning.concatenate(cls.wide_samples, 'mean')

    def test_initialization(self):
        self.assertEqual(self.scanning.hackrf.vga_gain, 0)
        self.assertEqual(self.scanning.hackrf.lna_gain, 0)
        self.assertEqual(self.scanning.hackrf.sample_rate, 20e6)
        self.assertEqual(self.scanning.overlap, self.scanning.overlap)

    def test_scanning(self):
        self.assertIsInstance(self.wide_samples, list)
        self.assertGreater(len(self.wide_samples), 0)
        for stay in self.wide_samples:
            self.assertIsInstance(stay, dict)
            self.assertIn('start', stay)
            self.assertIn('end', stay)
            self.assertIn('samples', stay)
            self.assertIn('sample_rate', stay)
            self.assertIn('overlap', stay)
            self.assertIsInstance(stay['start'], np.int64)
            self.assertIsInstance(stay['end'], np.int64)
            self.assertIsInstance(stay['samples'], np.ndarray)
            self.assertEqual(len(stay['samples']), self.scanning.samples_to_read)
            self.assertIsInstance(stay['sample_rate'], int)
            self.assertIsInstance(stay['overlap'], int)
            
    def test_concat(self):
        self.assertEqual(len(self.samples), self.scanning.samples_to_read*len(self.wide_samples)-(self.scanning.overlap_size*(len(self.wide_samples)-1)))

        
        



if __name__ == '__main__':
    unittest.main()
