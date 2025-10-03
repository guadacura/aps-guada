# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 10:58:19 2025

@author: USUARIO
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import fft, fftshift, fftfreq
#import scypy.stats as st 

fs = 1000
N = 1000
R = 200
fr = np.random.uniform(-2, 2, R)
deltaF = fs/N

SNR1 = 3
SNR2 = 10
a0 = 2
Omega0 = fs/4
ps = a0**2/2

pot_ruido1 = a0**2/(2*10**(SNR1/10))
pot_ruido2 = a0**2/(2*10**(SNR2/10))

#el reshape:
#Si tenías un vector de tamaño (N,) (unidimensional con N elementos),
#reshape((-1,1)) lo convierte en una matriz columna de forma (N,1).


#Matriz de senos
# Ventanas a usar
rectangular = windows.boxcar(N).reshape((-1,1))
hamming = windows.hamming(N).reshape((-1,1))
blackman_harris = windows.blackmanharris(N).reshape((-1,1))
flattop = windows.flattop(N).reshape((-1,1))

def ventaneo(x_ruido, vent):
    x_vent = x_ruido * vent
    X_vent = fft(x_vent, n=10*N, axis=0) / N
    return X_vent

tt_vector = np.arange(N)/fs
tt_columnas = tt_vector.reshape((-1,1))
ff_filas = fr.reshape((1,-1))
TT_sen = np.tile(tt_columnas, (1,R))
FF_sen = np.tile(ff_filas, (N,1))
ruido1 = np.random.normal(loc=0, scale=np.sqrt(pot_ruido1), size=(N,R))
ruido2 = np.random.normal(loc=0, scale=np.sqrt(pot_ruido2), size=(N,R))

xx_sen = a0 * np.sin(2 * np.pi * (Omega0 + FF_sen * deltaF) * TT_sen)

xx_sen_ruido1 = xx_sen + ruido1
xx_sen_ruido2 = xx_sen + ruido2

#-----------------------------------------
#Ventaneo normal sin zero padding
xx1_vent_rect=xx_sen_ruido1*rectangular
xx1_vent_ham=xx_sen_ruido1*hamming
xx1_vent_bh=xx_sen_ruido1*blackman_harris
xx1_vent_flat=xx_sen_ruido1*flattop

#---------------------------------------
#calculo FFT
XX1_rect = fft(xx1_vent_rect, axis=0)/N
XX1_ham = fft(xx1_vent_ham, axis=0)/N
XX1_bh = fft(xx1_vent_bh, axis=0)/N
XX1_flat = fft(xx1_vent_flat, axis=0)/N


#---------------------------------
#Datos que alimentan de los cuales se alimenta el histograma
idx = N//4  # índice en el tiempo donde tomar el histograma
datos = xx1_vent_rect[idx,:]  # muestras en ese instante para la ventana rectangular


plt.figure(figsize=(10,4))
plt.plot(datos, alpha=0.6, label='Muestras en t=N/4')
plt.axhline(0, color='k', linestyle='--')
plt.title("Muestras de ruido que alimentan el histograma")
plt.xlabel("Número de muestra (realización)")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid()
plt.show()
# -------------------------------
# Histograma en N/4

plt.figure(figsize=(10,6))
plt.hist(xx1_vent_rect[idx,:], bins=50, alpha=0.6, label='Rectangular')
plt.hist(xx1_vent_ham[idx,:], bins=50, alpha=0.6, label='Hamming')
plt.hist(xx1_vent_bh[idx,:], bins=50, alpha=0.6, label='Blackman-Harris')
plt.hist(xx1_vent_flat[idx,:], bins=50, alpha=0.6, label='Flattop')
plt.title("Histograma de ruido en t = N/4")
plt.xlabel("Amplitud [V]")
plt.ylabel("Frecuencia de ocurrencia")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# FFT (en dB) de la señal ventaneada con Hamming, por ejemplo
freqs = fftfreq(N, 1/fs)
X_ham_mean = np.mean(np.abs(XX1_ham), axis=1)  # promedio sobre realizaciones

plt.figure(figsize=(10,6))
plt.plot(freqs[:N//2], 20*np.log10(X_ham_mean[:N//2]))
plt.title("FFT (ventana Hamming) - Señal + Ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.grid()
plt.show()


#==============================================================
# Calcular FFTs con zero padding
XX_sen_ruido1 = fft(xx_sen_ruido1, n=10*N, axis=0) / N
XX_r1_rect = ventaneo(xx_sen_ruido1, rectangular)
XX_r1_ham = ventaneo(xx_sen_ruido1, hamming)
XX_r1_bh = ventaneo(xx_sen_ruido1, blackman_harris)
XX_r1_fl = ventaneo(xx_sen_ruido1, flattop)

XX_sen_ruido2 = fft(xx_sen_ruido2, n=10*N, axis=0) / N
XX_r2_rect = ventaneo(xx_sen_ruido2, rectangular)
XX_r2_ham = ventaneo(xx_sen_ruido2, hamming)
XX_r2_bh = ventaneo(xx_sen_ruido2, blackman_harris)
XX_r2_fl = ventaneo(xx_sen_ruido2, flattop)



# Frecuencias para el eje completo
Npad = 10 * N
freqs = np.arange(Npad) * fs / Npad   # de 0 a fs

plt.figure(figsize=(12, 8))
plt.plot(freqs, 10*np.log10(2*np.abs(XX_r1_rect[:,0])**2), label="Rectangular")
plt.plot(freqs, 10*np.log10(2*np.abs(XX_r1_ham[:,0])**2), label="Hamming")
plt.plot(freqs, 10*np.log10(2*np.abs(XX_r1_bh[:,0])**2), label="Blackman-Harris")
plt.plot(freqs, 10*np.log10(np.abs(XX_r1_fl[:,0])**2), label="Flattop")

plt.title("Estimación espectral unilateral (SNR = 3 dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.xlim(0, fs/2)   # <- solo hasta Nyquist
plt.grid(True)
plt.legend()
plt.show()

# ===============================
# Cálculo de estimadores en t=N/4
# ===============================

# Creamos un diccionario para las ventanas y sus datos en t=N/4
ventanas = {
    "Rectangular": xx1_vent_rect[idx,:],
    "Hamming": xx1_vent_ham[idx,:],
    "Blackman-Harris": xx1_vent_bh[idx,:],
    "Flattop": xx1_vent_flat[idx,:]
}

print("=== Estimadores estadísticos en t = N/4 ===")
for nombre, datos in ventanas.items():
    media = np.mean(datos)
    varianza = np.var(datos)
    desv_std = np.std(datos)
    minimo = np.min(datos)
    maximo = np.max(datos)
    print(f"\nVentana {nombre}:")
    print(f"  Media = {media:.5f} V")
    print(f"  Varianza = {varianza:.5e} V²")
    print(f"  Desviación estándar = {desv_std:.5f} V")
    print(f"  Rango = [{minimo:.5f}, {maximo:.5f}] V")





# plt.subplot(2, 1, 2)
# graficar(XX_sen_ruido2[:, 0], "Rectangular")
# graficar(XX_r2_ham[:, 0], "Hamming")
# graficar(XX_r2_bh[:, 0], "Blackman-Harris")
# graficar(XX_r2_fl[:, 0], "Flattop")
# plt.title("Estimación espectral con diferentes ventanas (SNR = 10 dB)")
# plt.xlabel("Frecuencia [múltiplos de Δf]")
# plt.ylabel("Magnitud [dB]")
# plt.ylim([-80, 5])
# plt.xlim(0, 5000)
# plt.grid(True)
# plt.legend()

plt.tight_layout()
plt.show()
