import numpy as np
import matplotlib.pyplot as plt
from detection import Detection
from processing import Processing

pros =Processing()
detect = Detection()

samples = np.load('tdt_5.npy')

f, Pxx = pros.welch(samples)
f = np.linspace(611, 631, len(Pxx))

index_c_i = np.where(np.isclose(f, 613, atol=0.01))[0][0]
index_c_s = np.where(np.isclose(f, 619, atol=0.01))[0][0]
index_c = np.where(np.isclose(f, 616, atol=0.01))[0][0]

freq_range = f[index_c_i:index_c_s + 1]
Pxx_range = Pxx[index_c_i:index_c_s + 1]

power = np.trapz(Pxx_range, freq_range)

index_n = np.where(np.isclose(f, 620, atol=0.01))[0][0]

c_n = (10*np.log10(power))/(10*np.log10(Pxx[index_n]))
print(c_n)

plt.semilogy(f, Pxx)
plt.semilogy(freq_range, Pxx_range)
plt.text(f[index_c], Pxx[index_c]+0.5*Pxx[index_c], f'C/N={c_n}', fontsize=12, ha='right')
#plt.ylim(5e-13, 8e-10)
plt.show()

