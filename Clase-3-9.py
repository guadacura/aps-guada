# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 21:27:39 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def mi_funcion_sen(ff, nn, fs, vmax=1, dc=0, ph=0):
    
    n = np.arange(0,nn)
    tt = n/fs
    w0 = 2 * np.pi * ff
    xx = dc + vmax * np.sin(w0 * tt + ph)
    
    return tt, xx
#%%
N = 1000
Fs = 1000
deltaf = Fs/N
t1,x1 = mi_funcion_sen(ff=250*deltaf,nn=N,fs=Fs)
t2,x2 = mi_funcion_sen(ff=251*deltaf,nn=N,fs=Fs)
t3,x3 = mi_funcion_sen(ff=250.5*deltaf,nn=N,fs=Fs)

X1 = fft(x1)
X2 = fft(x2)
X3 = fft(x3)
X1abs = np.abs(X1)
X2abs = np.abs(X2)
X3abs = np.abs(X3)

frec = np.arange(N) * deltaf
plt.figure(1)
plt.plot(frec,X1abs,':x',label="Transformada en N/4")
plt.plot(frec,X2abs,':x',label="Transformada en N/4+1")
plt.plot(frec,X3abs,':x',label="Transformada en N/4+1/2")
plt.xlim(0,Fs/2)
plt.title("FFT")
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Modulo de la señal')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(frec,np.log10(X1abs)*20,label="Transformada en N/4")
plt.plot(frec,np.log10(X2abs)*20,label="Transformada en N/4+1")
plt.plot(frec,np.log10(X3abs)*20,label="Transformada en N/4+1/2")
plt.xlim(0,Fs/2)
plt.title("FFT")
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid(True)
plt.show()

#%% actividad de parseval
N= 1000
fs=1000
deltaf=fs/N
tt,xx = mi_funcion_sen(ff=(N/4)*deltaf,nn=N,fs=fs) # se puede hacer definiendo vmax = raiz de 2
xp = xx /np.sqrt(np.var(xx))
print("Varianza:",np.var(xp))

Xf = np.fft.fft(xp)
Xmod = np.abs(Xf)**2
frec = np.arange(N) * deltaf
plt.figure(3)
plt.plot(frec,10*np.log10(Xmod))
plt.title("FFT")
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plt.show()


Et = np.sum(np.abs(xp)**2)
Efrec = (1/N)*np.sum(Xmod)
if Et == Efrec:
    print("Se cumple la identidad de Parseval")
else:
    print("Diferencia de la identidad de Parseval:",Et-Efrec)


Npad = 9 * N
xz = np.zeros(Npad)
xz[:N] = x3
Xz = fft(xz)
Xzabs = np.abs(Xz)
frec2 = np.arange(Npad) * Fs /Npad

plt.figure(4)
plt.plot(frec,X3abs,':x',label="Transformada en N/4+1/2")
plt.plot(frec2,Xzabs,label="Transformada en N/4+1/2 con zero-padding")
plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia Normalizada (×π rad/sample)')
plt.ylabel('Modulo de la señal')
plt.legend()
plt.grid(True)
plt.show()