# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 21:45:52 2025

@author: USUARIO
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


fm=1000 #frecuencia de muestreo
N=1000   #cantidad de muestras
dt=1/fm  #tiempo entre muestras
df=fm/N
fs=fm/4
#Seno
def funcion_seno(vc=1, dc=0, fs=None, ph=0, nn=N, fm=fm):
    t = np.linspace(0, N/fm, N, endpoint=False)
    sen=(vc*np.sin(2*np.pi*fs*t+ph)+dc)
    return t,sen
tt, yy = funcion_seno(vc=1, dc=0, fs=fm/4, ph=0, nn=N, fm=fm)



desvio=np.std(yy)
varianza=np.var(yy)

print(f'el desvio es {desvio} la varianza es {varianza}')

X1=(fft(yy))
X1abs=np.abs(X1)
X_dB = 10 * np.log10(X1abs**2) #densidad de potenciaaaa


# Energía en el tiempo
energia_tiempo = np.sum(np.abs(yy)**2)

# Energía en el dominio de la frecuencia (FFT)
energia_frecuencia = (1/N) * np.sum(X1abs**2)

print("Energía en tiempo:", energia_tiempo)
print("Energía en frecuencia:", energia_frecuencia)

ff= np.arange(N)*df
plt.figure(1)
plt.plot(ff,np.log10(X1abs)*20,'x', label='X1 abs en V' )
plt.title('Comparación ...')
plt.xlabel('Frecuencia (Hz)')
plt.xlim([0,fs/2])
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()





x2 = np.zeros(10*N) 
x2[:N] = yy # primera mitad senoidal, segunda mitad ceros

frec2 = np.arange(N*10) * fm /(N*10)


X2=(fft(x2))
X2abs=np.abs(X2)


plt.figure(1)
plt.plot(frec2,X2abs,'x', label='X2 abs en [dB]' )
plt.title('Comparación ...')
plt.xlim(240,260)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()
