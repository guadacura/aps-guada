# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 00:36:05 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt

#----------------------------
#esferización

aa=np.random.randn(10000) #genera numeros aleatorios con distribución normal estándar. Gauss
#la desviación estándar concetra la mayoría de sus valores entre [-3 sigma, 3sigma]
#sigma siendo el desvío estándar. que en este caso es 1
bb=np.random.randn(10000)

plt.plot(aa,bb,'x')

#---------------------------
#fuertemente coorrelacionada
plt.plot(aa,0.5*aa+1,'x')


#===================================
#welch y Blackman Tuckye¿
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

# ===============================
# 1) Parámetros de la señal
# ===============================
fs = 1000         # Frecuencia de muestreo [Hz]
N = 2048          # Cantidad de muestras
t = np.arange(N) / fs

f0 = 50           # Frecuencia de la senoidal
a0 = 1.0          # Amplitud
SNR_dB = 5        # SNR en dB

# ===============================
# 2) Señal + ruido
# ===============================
# Señal pura
x = a0 * np.sin(2 * np.pi * f0 * t)

# Potencia de la señal (media cuadrática)
Ps = np.mean(x**2)

# Potencia de ruido según SNR
Pn = Ps / (10**(SNR_dB/10))

# Genero ruido blanco gaussiano
ruido = np.random.normal(0, np.sqrt(Pn), size=N)

# Señal contaminada
x_ruidosa = x + ruido

# ===============================
# 3) Estimación espectral: BLACKMAN–TUCKEY
# ===============================
# Paso 1: Autocorrelación
rxx = np.correlate(x_ruidosa, x_ruidosa, mode='full')
lags = np.arange(-N+1, N)
rxx = rxx / N  # Normalizo

# Paso 2: Selecciono una ventana para la autocorrelación
M = 256  # long. truncamiento (<= N)
mid = len(rxx)//2
rxx_trunc = rxx[mid:mid+M]  # parte causal
ventana_bt = windows.hamming(M)  # Ventana para suavizar la autocorrelación
rxx_windowed = rxx_trunc * ventana_bt

# Paso 3: FFT de la autocorrelación para obtener la PSD
Rxx_BT = np.fft.fft(rxx_windowed, n=4096)
freqs_BT = np.fft.fftfreq(len(Rxx_BT), d=1/fs)
PSD_BT = np.abs(Rxx_BT)

# Me quedo con el semiespectro (frecuencias positivas)
mask_pos = freqs_BT >= 0
freqs_BT = freqs_BT[mask_pos]
PSD_BT = PSD_BT[mask_pos]

# ===============================
# 4) Estimación espectral: WELCH
# ===============================
f_welch, PSD_welch = welch(
    x_ruidosa,
    fs=fs,
    window='hamming',
    nperseg=256,
    noverlap=128,
    nfft=4096,
    scaling='density'
)

# ===============================
# 5) Graficación
# ===============================
plt.figure(figsize=(12,6))

# Señal en el tiempo
plt.subplot(2,1,1)
plt.plot(t, x_ruidosa, label="Señal ruidosa")
plt.plot(t, x, 'r--', alpha=0.6, label="Señal pura")
plt.xlim([0, 0.1])  # solo muestro un fragmento
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.title("Señal senoidal con ruido")
plt.legend()
plt.grid()

# PSDs
plt.subplot(2,1,2)
plt.semilogy(freqs_BT, PSD_BT, label='Blackman–Tukey')
plt.semilogy(f_welch, PSD_welch, label='Welch')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V²/Hz] (escala log)")
plt.title("Estimación espectral (PSD)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
