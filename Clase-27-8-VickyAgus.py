# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 16:46:58 2025

@author: USUARIO
"""
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

fs = 1000
ts = 1/fs
N = 1000
nn = 1000
tt = np.arange(nn)*ts

#FFT de señal senoidal 

x = np.sin(tt*np.pi*2*fs/4)

transf = np.fft.fft(x)

#print(transf)

#FFT de señal adyacente

y = np.sin(2* np.pi* (fs/ 4 + fs/nn)*tt)

transf_2 = np.fft.fft(y)

#FFT de señal en el medio de ambas

a = np.sin(2*np.pi*((fs/4)+(fs/(2*nn)))*tt)

transf_3 = np.fft.fft(a)

plt.figure(figsize=(10,4))
plt.plot( np.abs(transf), '--', label="Transformada 1")
plt.plot(np.abs(transf_2), label="Transformada 2(adyacente)")
plt.plot(np.abs(transf_3), '-.', label="Transformada 3(punto medio)")
plt.legend()
plt.grid(True)
plt.xlabel("Muestras")
plt.title("FFT 1")
plt.tight_layout()
plt.show()


Npad = 9 * N
xz = np.zeros(Npad)
xz[:N] = x
Xz = fft(xz)
Xzabs = np.abs(Xz)
frec2 = np.arange(Npad) * fs /Npad

