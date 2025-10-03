# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:42:19 2025

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

# Ventanas a analizar
ventanas = {
    "Blackman Harris": "blackmanharris",
    "Hamming": "hamming",
    "Hann": "hann",
    "Rectangular": "boxcar",
    "Flattop": "flattop"
}

N = 31  # tamaño de la ventana
M = 2048  # número de puntos para la FFT
delta_f = 1/ M

plt.figure(figsize=(10,5))

for nombre, tipo in ventanas.items():
    w = get_window(tipo, N)
    
    # FFT y centrado en cero
    W = np.fft.fft(w, M)
    W = np.fft.fftshift(W)
    
    # Eje de frecuencias de -pi a pi
    freqs = np.linspace(-0.5, 0.5, M) / delta_f  # en múltiplos de Δf
    
    # Magnitud en dB normalizada
    W_dB = 20 * np.log10(np.abs(W) / np.max(np.abs(W)))
    
    plt.plot(freqs, W_dB, label=nombre)




# Ajustes del gráfico
plt.title("Ventanas")
plt.ylabel(r"$|W_N(\omega)|$ [dB]")
plt.xlabel("Frecuencia en multiplos de Δf")
plt.ylim(-80, 0)
plt.xlim(-np.pi, np.pi)
plt.legend()
plt.grid()
plt.show()

#----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

# Señal de ejemplo (puedes reemplazarla por tus datos)
fs = 1000  # frecuencia de muestreo en Hz
t = np.arange(0, 1, 1/fs)  # 1 segundo
f0 = 50   # Hz
signal = np.sin(2*np.pi*f0*t) + 0.3*np.sin(2*np.pi*120*t)  # señal con 2 senoidales

# Ventanas a aplicar
ventanas = {
    "Hamming": "hamming",
    "Blackman Harris": "blackmanharris",
    "Flattop": "flattop"
}

plt.figure(figsize=(12,6))

for nombre, tipo in ventanas.items():
    w = get_window(tipo, len(signal))
    señal_vent = signal * w
    fft_win = np.fft.fft(señal_vent)
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    plt.plot(freqs[:len(freqs)//2],
             20*np.log10(np.abs(fft_win[:len(freqs)//2]) / np.max(np.abs(fft_win))),
             label=nombre)

plt.title("Espectro de la señal con distintas ventanas")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim(-100, 0)
plt.xlim(0, 200)  # mostrar solo bajas frecuencias
plt.legend()
plt.grid()
plt.show()

