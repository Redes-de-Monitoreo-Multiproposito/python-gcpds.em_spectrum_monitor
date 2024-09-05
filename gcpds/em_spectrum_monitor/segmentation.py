import os
import numpy as np
import pandas as pd 
from monitor import Scanning
from processing import Processing
from detection import Detection
import matplotlib.pyplot as plt

class RDS():
    """
    RDS class is responsible for measure the parameters of emitting abote the broadcasters
    """
    def __init__(self):
        """
        Initializes the RDS class with instances of Processing and Detection
        and sets up attributes for later use.
        """
        self.pros = Processing()
        self.detec = Detection()
    
    def broadcasters(self, town: str = 'Manizales'):
        """
        Filters and extracts FM radio frequencies for a given town from a CSV file.

        Parameters
        ----------
            town : str
                The town for filter the broadcasters
        
        Returns:
            None
        """
        df = pd.read_csv('Radioemisoras Colombia - Radioemisoras 2023.csv')

        datos_filtrados = df[(df['Municipio'].str.upper() == town.upper()) & 
                                (df['Tecnología transmisión'] == 'FM')]
    
        frequencies = datos_filtrados['Frecuencia'].str.replace(' MHz', '', regex=False).astype(float)
        self.frequencies = sorted(frequencies)

    def segmentation(self):
        """
        This method performs the segmentation of the specified samples between presence and non-presence 
        and is stored in lists with their respective label 1 or 0

        Parameters
        ----------
            NONE
        
        Returns:
            presence: list with parameters freq, Pxx and label for presence.
            ausence: list with parameters freq, Pxx and label for ausence.
        """

        if not os.path.exists('segmentation'):
            os.makedirs('segmentation')

        presence = []
        ausence = []
        # freq_full_n = []
        # Pxx_full_n = []
        # freq_full_p = []
        # Pxx_full_p = []

        for i in range(1800):
            samples = np.load(f'database_prueba_piloto/Samples 88 and 108MHz, time to read 0.01s, sample #{i}.npy')
            print(f'Segmenting sample #{i}')

            f, Pxx = self.pros.welch(samples, 20e6)
            f = np.linspace(88, 108, len(Pxx))
            noise = np.arange(92.05, 94.3, 0.25)

            for center_freq in self.frequencies:
                lower_index = np.argmin(np.abs(f - (center_freq - 0.125)))
                upper_index = np.argmin(np.abs(f - (center_freq + 0.125)))

                freq_range = f[lower_index:upper_index + 1]
                Pxx_range = Pxx[lower_index:upper_index + 1]

                presence.append({
                    'freq': freq_range,
                    'Pxx': Pxx_range,
                    'label': 1})
                
                # freq_full_p.append(freq_range)
                # Pxx_full_p.append(Pxx_range)
                
            for center_freq in noise:
                lower_index = np.argmin(np.abs(f - (center_freq - 0.125)))
                upper_index = np.argmin(np.abs(f - (center_freq + 0.125)))

                freq_range = f[lower_index:upper_index + 1]
                Pxx_range = Pxx[lower_index:upper_index + 1]

                ausence.append({
                    'freq': freq_range,
                    'Pxx': Pxx_range,
                    'label': 0
                })

                # freq_full_n.append(freq_range)
                # Pxx_full_n.append(Pxx_range)

            print(f'Samples #{i} segmented')
            # plt.semilogy(f, Pxx)
            # for i in range(len(freq_full_n)):
            #     print(i)
            #     plt.semilogy(freq_full_n[i], Pxx_full_n[i], c='red', linestyle='dashdot')
            #     plt.semilogy(freq_full_p[i], Pxx_full_p[i], c='yellow', linestyle='dashdot')
            # plt.show()
        return presence, ausence