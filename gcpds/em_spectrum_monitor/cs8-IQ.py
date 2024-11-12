import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

# Cargar el archivo .cs8 en formato int8
nombre_archivo = 'gcpds/em_spectrum_monitor/test.cs8'
datos = np.fromfile(nombre_archivo, dtype=np.int8)

# Separar las muestras en componentes I y Q, asumiendo que los datos están intercalados
I = datos[::2]    # Componentes I (pares)
Q = datos[1::2]   # Componentes Q (impares)

# Crear señal compleja a partir de I y Q
signal = I + 1j * Q

# signal = np.load('database/Samples 88 and 108MHz,time to read 0.01s, sample #0.npy')

f, Pxx = sig.welch(signal, 20, nperseg=1024, window='hann')
f = np.fft.fftshift(f)
f = np.linspace(88, 108, len(Pxx))
Pxx = np.fft.fftshift(Pxx)

# Graficar el espectro de frecuencia
plt.plot(f, 10*np.log10(Pxx))
plt.title('Espectro de Frecuencia')
plt.xlabel('Frecuencia')
plt.ylabel('Magnitud')
plt.show()
