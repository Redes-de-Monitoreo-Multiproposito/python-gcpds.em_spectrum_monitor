import asyncio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import welch
from hackrf.scan import ScanHackRF
from processing import Processing
from detection import Detection
import time

detec = Detection()
pros = Processing()
broad = detec.broadcasters('Manizales')

count = 0


parameters = []
parameters_1s = []
df_12h = pd.DataFrame()

# ----------------------------------------------------------------------
def custom_callback(data_freqs, sweep_config):
    """"""
    global count
    if count < 60:
        count += 1
        f, Pxx = pros.welch(np.array(data_freqs[88000000]), fs=sweep_config['sample_rate'])
        f1, Pxx1 = plt.psd(np.array(data_freqs[88000000]), 1024, 20, 98)
        f = np.linspace(88, 108, len(Pxx))

        plt.semilogy(f, Pxx)

        _, _, threshold = detec.power_based_detection(f, Pxx)

        # comparation = np.any(np.abs(peak_freqs[:, None] - broad) < 0.1, axis=1)

        # no_reg = peak_freqs[~comparation]

        t0 = time.strftime('%X')

        # for i in range(len(no_reg)):
        #     f_start, f_end = detec.bandwidth(f, Pxx, no_reg[i], noise_lvl)
        #     bandwidth = f_end - f_start
        #     index = np.where(np.isclose(f, no_reg[i], atol=0.01))[0]
        #     parameters.append({
        #                 'time': t0,
        #                 'freq': round(no_reg[i], 1),
        #                 'bandwidth': round(bandwidth, 2),
        #                 'power': Pxx[index[0]],
        #                 'snr': 10 * np.log10(Pxx[index[0]]/Pxx[0]),
        #                 'registered': 'no' 
        #             })
        #     plt.axvline(f_start, c='red')
        #     plt.axvline(f_end, c='red')

        # for i in range(len(broad)):
        #     f_start, f_end = detec.bandwidth(f, Pxx, broad[i], 2e-15)
        #     bandwidth = f_end - f_start
        #     index = np.where(np.isclose(f, broad[i], atol=0.01))[0]

        #     parameters.append({
        #                 'time': t0,
        #                 'freq': round(broad[i], 1),
        #                 'bandwidth': round(bandwidth, 2),
        #                 'power': Pxx[index[0]],
        #                 'snr': 10 * np.log10(Pxx[index[0]]/Pxx[0]),
        #                 'registered': 'yes'
        #             })
        #     plt.axvline(f_start, c='red')
        #     plt.axvline(f_end, c='red')

        # plt.scatter(peak_freqs, peak_powers)
        # plt.axhline(noise_lvl, c='green')
        plt.axhline(threshold, c='green')
        plt.show()
        parameters_1s.append(parameters)
        print(f'Muestra #{count} adquirida y procesada')

        if len(parameters_1s) >= 5:
            data = [item for sublist in parameters_1s for item in sublist]
            
            df_5m = pd.DataFrame(data)

            df_5m = df_5m.groupby('freq', as_index=False).agg({
                'bandwidth': 'mean',
                'power': 'mean',
                'snr': 'mean',
                'time': 'last',
                'registered': 'last'
            })

            global df_12h 
            df_12h = pd.concat([df_12h, df_5m], ignore_index=True)

            parameters_1s.clear()
            
            print(f'Promedio procesado')

        if count == 60:     
            df_12h.to_excel('RDS.xlsx', index=False)
            df_12h = pd.DataFrame()

        # time.sleep(1)

async def main():
    scanhackrf = ScanHackRF(0)
    scanhackrf.vga_gain = 0
    scanhackrf.lna_gain = 0

    await scanhackrf.scan(
        bands=[
            # List of frequency tuples (start_freq, end_freq) in MHz
            (88, 108)
        ],
        sample_rate=20e6,  # Sample rate in samples per second (20 million samples per second)
        step_width=20e6,  # Frequency step width in Hz (10 MHz)
        step_offset=None,  # Step offset in Hz (5 MHz from the target frequency)
        read_num_blocks=9,  # Number of blocks to read in each step
        buffer_num_blocks=20,  # Number of blocks to buffer
        callback=custom_callback,  # Optional callback function to process the data
        interleaved=False,
    )

asyncio.run(main())