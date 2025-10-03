# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 17:54:58 2025

@author: USUARIO
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import fft, fftshift
from scipy import signal as sig
import scipy.io as sio
from scipy.io.wavfile import write


fs=1000 #en Hz
N=1000 #muestras
df=fs/N #resolucion espectral,en Hz
ts=1/fs #período de muestreo en segundos
Nyquist=fs/2

n=np.arange(N)
t=n/fs

dt = 1/fs
#energia = np.sum(yy**2)*dt




##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure()
plt.plot(ecg_one_lead[5000:12000])

plt.figure()
plt.plot(hb_1)

plt.figure()
plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure()
plt.plot(ecg_one_lead)

#%%
#==================================
#ESTIMO CON WELCH
N=(ecg_one_lead.shape[0])
cantidadPromedios=30
nperseg=N//cantidadPromedios

flattop = windows.flattop(nperseg)
hamming = windows.hamming((nperseg))
blackmanharris = windows.blackmanharris((nperseg))

f_welch_f, Pxx_welch1=sig.welch(ecg_one_lead,fs=fs, window=flattop, nperseg=nperseg, nfft=4*nperseg)
f_welch_h, Pxx_welch2=sig.welch(ecg_one_lead,fs=fs, window=hamming)
f_welch_bh, Pxx_welch3=sig.welch(ecg_one_lead,fs=fs, window=blackmanharris)


plt.figure(figsize=(10,5))
plt.plot(f_welch_f, Pxx_welch1,label='Welch con flattop')
plt.title("PSD estimada con Welch")
plt.xlim(0,50)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V²/Hz]")
plt.grid(True)
plt.show()
