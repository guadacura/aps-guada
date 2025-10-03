# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:28:47 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt

N = 4000
ff = 4000  # frecuencia de muestreo (>= 2*fs)
vc=1
#vc=amplitud max 
#dc=desfasaje-valor medio 
#ff=frecuencia de muestreo 
#ph=fase inicial. Es el desplazamiento horizontal de la señal en radianes. 
#fs=frecuencia de señal

def funcion_seno(vc=None, dc=None, fs=None, ph=None, nn=N, ff=ff):
    t = np.arange(nn) / ff
    sen = vc * np.sin(2 * np.pi * fs * t + ph) + dc
    return t, sen

def funcion_seno_modulada(vc=None, dc=None, fs=None, ph=None, nn=N, ff=ff,yy1=None):
    t = np.arange(nn) / ff
    sen = (vc+ yy1) * np.sin(2 * np.pi * fs * t + ph) + dc
    return t, sen

fs = 2000

tt, yy = funcion_seno(vc=vc, dc=0, fs=fs, ph=0, nn=N, ff=ff)

plt.plot(tt, yy, label=f'{fs} Hz')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencia")
plt.grid(True)
plt.show()

tt, yy = funcion_seno(vc=np.pi/2, dc=0, fs=fs, ph=np.pi/2, nn=N, ff=ff)

plt.plot(tt, yy, label=f'{fs} Hz')
plt.title('Señal Senoidal desplazada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencia")
plt.grid(True)
plt.show()


tt1,yy1=funcion_seno(vc=np.pi/2, dc=0, fs=2000/2, ph=np.pi/2, nn=N, ff=ff)
tt, yy= funcion_seno_modulada(vc=np.pi/2, dc=0, fs=2000/2, ph=np.pi/2, nn=N, ff=ff,yy1=yy1)

plt.plot(tt, yy, label=f'{fs} Hz')
plt.title('Señal Senoidal desplazada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.legend(title="Frecuencia")
plt.grid(True)
plt.show()


#Cuadrado de la amplitud es la potencia
#Potencia original= A^2
#Si quiero el 75%, =pot original*0,75 

AmplitudNueva=np.sqrt(0.75*(vc**2))

tt,yy=funcion_seno(vc=AmplitudNueva, dc=0, fs=2000/2, ph=np.pi/2, nn=N, ff=ff)