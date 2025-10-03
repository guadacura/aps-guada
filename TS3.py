# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 16:08:19 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

N=1000
fs=1000
df=fs/N #espaciamiento de frecuencia en la DFT
ts=1/fs

#grafico
def funcion_grafico(x, y, frecuencia, titulo, ejex, ejey):  
    plt.plot(x, y, label=f'{frecuencia} Hz')
    plt.title(titulo)
    plt.xlabel(ejex)
    plt.ylabel(ejey)
    plt.legend(title="Frecuencia")
    plt.grid(True)
    plt.show()
    
    
def sen(ff,nn,amp=np.sqrt(2), dc=0, ph=0, fs=2):
    n=np.arange(nn)
    t=n/fs
    x=dc+amp*np.sin(2*np.pi*ff*t+ph)
    return t,x

t1,x1= sen(ff=(N/4)*df, nn=N,amp=np.sqrt(2), fs=fs)
t2,x2= sen(ff=((N/4)+0.25)*df, amp=np.sqrt(2), nn=N, fs=fs)   
t3,x3= sen(ff=((N/4)+0.5)*df, nn=N, amp=np.sqrt(2), fs=fs)   

# Verificar potencia unitaria
print(f"Varianza x1: {np.var(x1):.6f}")
print(f"Varianza x2: {np.var(x2):.6f}") 
print(f"Varianza x3: {np.var(x3):.6f}")

#---------------------------------------------------
#gráfico de las senoidales normales

#plt.figure(1)
#funcion_grafico(t1, x1,'(N/4)*df', 'Senoidal con frecuencia (N/4)*Δf','tiempo[s]', 'Amplitud[V]')

#plt.figure(2)
#funcion_grafico(t2, x2,'(N/4)+0.25*df' , 'Senoidal con frecuencia [(N/4)+0,25]*Δf','tiempo[s]', 'Amplitud[V]')

#plt.figure(3)
#funcion_grafico(t3, x3, '(N/4)+0.5*df' , 'Senoidal con frecuencia [(N/4)+0,5]*Δf','tiempo[s]', 'Amplitud[V]')

#-------------------------------------------------
#Calculo de la DFT -> FFT

X1=(fft(x1))
X2=(fft(x2))
X3=(fft(x3))

#calculo valor abs
X1abs=np.abs(X1)
X2abs=np.abs(X2)
X3abs=np.abs(X3)

#defino eje de frecuencias
ff= np.arange(N)*df

#Desparramo espectral
#PDS
pds1=(2/(N**2))*X1abs**2
pds2=(2/(N**2))*X2abs**2
pds3=(2/(N**2))*X3abs**2


plt.figure(2)
plt.plot(ff,10*np.log10(pds1),label="PDS para f=N/4")
plt.plot(ff,10*np.log10(pds2),label="PDS para f=N/4+0,25")
plt.plot(ff,10*np.log10(pds3),label="PDS para f=N/4+0,5")
plt.xlim(230,270)
plt.title("PDS")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PDS [dB]')
plt.legend()
plt.grid(True)
plt.show()

#plt.figure(4)
#funcion_grafico(ff,np.log10(X1abs)*20,'(N/4)*df','FFT De las senoidales', 'frecuencia [Hz]', 'Amplitud [dB]' )

#plt.figure(5)
#funcion_grafico(ff,np.log10(X2abs)*20,'(N/4)+0.25*df','FFT De las senoidales', 'frecuencia [Hz]', 'Amplitud [dB]' )

#plt.figure(6)
#funcion_grafico(ff,np.log10(X3abs)*20,'(N/4)+0.5*df','FFT De las senoidales', 'frecuencia [Hz]', 'Amplitud [dB]' )

#--------------------------------------------------
#Parseval
#1)

xp1 = x1 /np.sqrt(np.var(x1))
print("Varianza:",np.var(xp1))

Xf = np.fft.fft(xp1)
Xmod = np.abs(Xf)**2


Et = np.sum(np.abs(xp1)**2)
Efrec = (1/N)*np.sum(Xmod)
if Et == Efrec:
    print("Para el primero: \nSe cumple la identidad de Parseval")
else:
    print("Para el primero: \nDiferencia de la identidad de Parseval:",Et-Efrec)
    
    
#2) 
xp2 = x2 /np.sqrt(np.var(x2))
print("Varianza:",np.var(xp2))

Xf = np.fft.fft(xp2)
Xmod = np.abs(Xf)**2


Et = np.sum(np.abs(xp2)**2)
Efrec = (1/N)*np.sum(Xmod)
if Et == Efrec:
    print("Para el segundo: \nSe cumple la identidad de Parseval")
else:
    print("Para el segundo: \nDiferencia de la identidad de Parseval:",Et-Efrec)
 
    
#3)
xp3 = x3 /np.sqrt(np.var(x3))
print("Varianza:",np.var(xp3))

Xf = np.fft.fft(xp3)
Xmod = np.abs(Xf)**2


Et = np.sum(np.abs(xp3)**2)
Efrec = (1/N)*np.sum(Xmod)
if Et == Efrec:
    print("Para el tercero: \nSe cumple la identidad de Parseval")
else:
    print("Para el tercero: \nDiferencia de la identidad de Parseval:",Et-Efrec)

#------------------------------------------------------
#zero padding

#1)
Npad = 10 * N
xz1 = np.zeros(Npad)
xz1[:N] = x1
Xz1 = fft(xz1)
Xz1abs = np.abs(Xz1)
frec2 = np.arange(Npad) * fs /Npad

plt.figure(4)
plt.plot(ff,np.log10(X1abs)*20,':x',label="Transformada en N/4")
plt.plot(frec2,np.log10(Xz1abs)*20,label="Transformada en N/4 con zero-padding")
plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid(True)
plt.show()

#2)
xz2 = np.zeros(Npad)
xz2[:N] = x2
Xz2 = fft(xz2)
Xz2abs = np.abs(Xz2)
frec2 = np.arange(Npad) * fs /Npad

plt.figure(5)
plt.plot(ff,np.log10(X2abs)*20,':x',label="Transformada en N/4+0,25")
plt.plot(frec2,np.log10(Xz2abs)*20,label="Transformada en N/4+0,25 con zero-padding")
plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid(True)
plt.show()

#3)
xz3 = np.zeros(Npad)
xz3[:N] = x3
Xz3 = fft(xz3)
Xz3abs = np.abs(Xz3)
frec2 = np.arange(Npad) * fs /Npad

plt.figure(6)
plt.plot(ff,np.log10(X3abs)*20,':x',label="Transformada en N/4+0,5")
plt.plot(frec2,np.log10(Xz3abs)*20,label="Transformada en N/4+0,5 con zero-padding")
plt.xlim(240,260)
plt.title("FFT")
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------------
#Bonus

