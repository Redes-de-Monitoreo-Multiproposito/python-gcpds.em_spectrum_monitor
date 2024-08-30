import time
import os
import numpy as np
import pandas as pd 
from monitor import Scanning
from processing import Processing
from detection import Detection

class RDS():

    def __init__(self, 
                 vga_gain: int = 0,
                 lna_gain: int = 0,
                 sample_rate: float = 20e6,
                 overlap: int = 0,
                 time_to_read: float = 1,):
        self.scan = Scanning(vga_gain=vga_gain, lna_gain=lna_gain, sample_rate=sample_rate, overlap=overlap, time_to_read=time_to_read)
        self.pros = Processing()
        self.detec = Detection()
        """Initialize the RDS object with the given parameters.

        Parameters
        ----------
        vga_gain : int, optional
            Gain for the VGA (default is 0).
        lna_gain : int, optional
            Gain for the LNA (default is 0).
        sample_rate : float, optional
            Sample rate in Hz (default is 20e6).
        overlap : int, optional
            Overlap between frequency steps in Hz (default is 0).
        time_to_read : float, optional
            Duration of time to read samples in seconds (default is 1).
        """
    
    def broadcasters(self, town: str = 'Manizales'):
        """The frequencies where there are radio stations are extracted according to the selected city.

        Parameters
        ----------
        town : string
            Town to extract broadcasters

        Returns
        -------
        NONE
        """
        df = pd.read_csv('Radioemisoras Colombia - Radioemisoras 2023.csv')

        datos_filtrados = df[(df['Municipio'].str.upper() == town.upper()) & 
                                (df['Tecnología transmisión'] == 'FM')]
    
        frequencies = datos_filtrados['Frecuencia'].str.replace(' MHz', '', regex=False).astype(float)
        self.frequencies = sorted(frequencies)

    def parameter(self, hours_to_scan: int = 1, save: bool = True):
        """The bandwidth and maximum power parameters of each radio station are calculated every second. 
        Every 5 minutes, all the obtained parameters are averaged, and at the end of the analysis hours, 
        the averages are averaged again.

        Parameters
        ----------
        hours_to_scan : int
            Total hours to analyze
        save : bool
            Selection parameters for saving the samples

        Returns
        -------
        parameters_prom_12h_final: dictionarie contains:
        - 'freq': float : Frequencies for each broadcaster.
        - 'bandwidth' : Bandwidth for each broadcaster.
        - 'power' : power in central frequency for each broadcaster.

        parameters_prom_12h: dictionarie of list contains:
        - 'freq': float : Frequencies for each broadcaster.
        - 'bandwidth' : Bandwidth for each broadcaster.
        - 'power' : power in central frequency for each broadcaster.
        """

        if not os.path.exists('database_prueba_piloto'):
            os.makedirs('database_prueba_piloto')

        if not os.path.exists('database_prueba_piloto_h5'):
            os.makedirs('database_prueba_piloto_h5')

        samples = np.load('database/Samples 88.0 and 108.0MHz with time to read 0.01s and 0MHz overlap.npy')
        
        parameters_prom_5m = []
        parameters_prom_12h = []
        times = (hours_to_scan * 60 * 60)

        for i in range(int(times)):
            
            wide_samples = self.scan.scan(88e6, 108e6)
            samples = self.scan.concatenate(wide_samples, 'mean') 

            if save == 1:
                np.save(os.path.join('database', f'Samples 88 and 108MHz,time to read 0.01s, sample #{i}.npy'), samples)

            f, Pxx = self.pros.welch(samples, 20e6)
            f  = np.linspace(88, 108, len(Pxx))
            
            parameters = []

            for j in range(len(self.frequencies)):
                f_start, f_end = self.detec.bandwidth(f, Pxx, self.frequencies[j])
                bandwidth = f_end - f_start
                    
                parameters.append({
                    'freq': round(self.frequencies[j], 1),
                    'bandwidth': round(bandwidth, 2),
                    'power': Pxx[j]
                })

            parameters_prom_5m.append(parameters)
            print(f'Muestra #{i} adquirida')
            time.sleep(1)

            if (i + 1) % 300 == 0:

                suma_freq = np.zeros(len(parameters_prom_5m[0]))
                suma_bandwidth = np.zeros(len(parameters_prom_5m[0]))
                suma_power = np.zeros(len(parameters_prom_5m[0]))

                for group in parameters_prom_5m:
                    for item in group:          
                        index = group.index(item)
                        suma_freq[index] += item['freq']
                        suma_bandwidth[index] += item['bandwidth']
                        suma_power[index] += item['power']
                
                num_items = len(parameters_prom_5m)
                prom_freq = [suma / num_items for suma in suma_freq]
                prom_bandwidth = [round(suma, 2) / num_items for suma in suma_bandwidth]
                prom_power = [suma / num_items for suma in suma_power]

                parameters_prom_5m_avg= ({
                    'freq': prom_freq,
                    'bandwidth': prom_bandwidth,
                    'power': prom_power
                })
                parameters_prom_5m = []
                parameters_prom_12h.append(parameters_prom_5m_avg)
                print(f'Promedio {i/300} adquirido')

        num_items = len(parameters_prom_12h)
        suma_freq = np.zeros(len(parameters_prom_12h[0]['freq']))
        suma_bandwidth = np.zeros(len(parameters_prom_12h[0]['bandwidth']))
        suma_power = np.zeros(len(parameters_prom_12h[0]['power']))

        for group in parameters_prom_12h:
            for idx in range(len(group['freq'])):          
                suma_freq[idx] += group['freq'][idx]
                suma_bandwidth[idx] += group['bandwidth'][idx]
                suma_power[idx] += group['power'][idx]
                
        prom_freq = [suma / num_items for suma in suma_freq]
        prom_bandwidth = [round(suma, 2) / num_items for suma in suma_bandwidth]
        prom_power = [suma / num_items for suma in suma_power]

        parameters_prom_12h_final = {
            'freq': prom_freq,
            'bandwidth': prom_bandwidth,
            'power': prom_power
        }
        print(f'Promedio total adquirido')
        return parameters_prom_12h_final, parameters_prom_12h