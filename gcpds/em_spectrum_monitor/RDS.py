import time
import threading
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
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
        #self.scan = Scanning(vga_gain=vga_gain, lna_gain=lna_gain, sample_rate=sample_rate, overlap=overlap, time_to_read=time_to_read)
        self.pros = Processing()
        self.detec = Detection()
        self.excel_file = 'Hoja de cálculo sin título (1).xlsx'
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

    def save_to_excel(self, df, sheet_name='Hoja 1'):
        df = pd.DataFrame([df])  # Convertir el dato en un DataFrame de pandas
        try:
            with pd.ExcelWriter(self.excel_file, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False, startrow=writer.sheets[sheet_name].max_row)
        except FileNotFoundError:
            df.to_excel(self.excel_file, sheet_name=sheet_name, index=False)

    def parameter(self, hours_to_scan: int = 1, city: str = 'Manizales'):
        """The bandwidth and maximum power parameters of each radio station are calculated every second. 
        Every 5 minutes, all the obtained parameters are averaged, and at the end of the analysis hours, 
        the averages are averaged again.

        Parameters
        ----------
        hours_to_scan : int
            Total hours to analyze
        city : str
            

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

        frequencies = self.detec.broadcasters(city)
        
        parameters_1s = []
        parameters = []
        df_12h = pd.DataFrame()
        times = (hours_to_scan * 60)

        # f, Pxx = pros.welch(np.array(data_freqs[88000000]), fs=sweep_config['sample_rate'])
        # f, Pxx = self.pros.welch(samples, fs=20e6)
        # f = np.linspace(88, 108, len(Pxx))

        for i in range(int(times)):
            
            t0 = time.time()
            samples = np.load(f'database/Samples 88 and 108MHz,time to read 0.01s, sample #{i}.npy')

            f, Pxx = self.pros.welch(samples, fs=20e6)
            f = np.linspace(88, 108, len(Pxx))

            _, _, noise_lvl = self.detec.power_based_detection(f, Pxx)            
            # plt.semilogy(f, Pxx)
            
            for j in range(len(frequencies)):

                f_start, f_end = self.detec.bandwidth(f, Pxx, frequencies[j], noise_lvl)
                bandwidth = f_end - f_start

                index = np.where(np.isclose(f, frequencies[j], atol=0.01))[0]

                parameters.append({
                            'time': time.strftime('%X'),
                            'freq': round(frequencies[j], 1),
                            'bandwidth': round(bandwidth, 2),
                            'power': Pxx[index[0]],
                            'snr': 10 * np.log10(Pxx[index[0]]/Pxx[0])
                        })
                plt.axvline(f_end, c='red')
                plt.axvline(f_start, c='red')
            parameters_1s.append(parameters)
        
            # plt.show()

            print(f'Muestra min {i+1} adquirida y procesada')

            if len(parameters_1s) >= 5:
                data = [item for sublist in parameters_1s for item in sublist]
                
                df_5m = pd.DataFrame(data)

                df_5m = df_5m.groupby('freq', as_index=False).agg({
                    'bandwidth': 'mean',
                    'power': 'mean',
                    'snr': 'mean',
                    'time': 'last'
                })

                df_12h = pd.concat([df_12h, df_5m], ignore_index=True)

                parameters_1s.clear()

                print(f'Promedio {(i+1)/5} adquirido')
            # time.sleep(60)
            print(f'Tiempo: {time.time()-t0}')
    
        # df_12h.to_excel('RDS.xlsx', index=False)
        print(df_12h.head(20))
        df_12h = pd.DataFrame()
        plt.semilogy(f, Pxx)
        plt.axhline(noise_lvl, c='green')
        plt.show()

        return df_12h