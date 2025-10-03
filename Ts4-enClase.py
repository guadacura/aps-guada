# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:45:34 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import fft, fftshift

#senoidal normal con fs=N/4

fs=1000 #en Hz
N=1000 #muestras
df=fs/N #resolucion espectral,en Hz

amp=np.sqrt(2) #es para normalizar a 1 la potencia, en Volts

def sen(ff,nn,amp=amp, dc=0, ph=0, fs=2):
    n=np.arange(nn)
    t=n/fs
    x=dc+amp*np.sin(2*np.pi*ff*t+ph)
    return t,x

t1,s1= sen(ff=(fs/4), nn=N, fs=fs)

S1=fft(s1)
S1abs=np.abs(S1)

ff= np.arange(N)*df
plt.figure(1)
plt.plot(ff,np.log10(S1abs)*20, label='X1 abs en V' )
plt.title('FFT de la senoidal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()

#----------------------------------------
SNR1=3 #en dB
SNR2=10 #en dB

pot_ruido1=amp**2/(2*10**(SNR1/10))
pot_ruido2=amp**2/(2*10**(SNR2/10))

print(f'La potencia del ruido para SNR=3 es {pot_ruido1:.3f}')
print(f'La potencia del ruido para SNR=10 es {pot_ruido2:.3f}')

ruido1=np.random.normal(0,np.sqrt(pot_ruido1),N) #pasas la raiz de la varianza ->desvío estandar
ruido2=np.random.normal(0,np.sqrt(pot_ruido2),N)

var_ruido1=np.var(ruido1)
var_ruido2=np.var(ruido2)

print(f'La potencia de ruido (varianza) para SNR=3 es {var_ruido1:.3f}')
print(f'La potencia del ruido (varianza) para SNR=10 es {var_ruido2:.3f}')

#------------------------------------------------
#Hago señal senoidal con ruido, la llamo x
x1=s1+ruido1
x2=s1+ruido2

X1=fft(x1)
X2=fft(x2)

X1abs=np.abs(X1)
X2abs=np.abs(X2)

plt.figure(2)
plt.plot(ff,20*np.log10(X1abs), label='X1(SNR=3) abs en V' )
plt.plot(ff,20*np.log10(X2abs), label='X2(SNR=10) abs en V' )
plt.title('FFT de la senoidal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()

#lo que se ve es un aumento en el ruido, de S1 a X1 y X2

#-----------------------------------------------------
#veo que la suma entre la senoida, más el ruido, da la funcion ruidosa para SNR=10

#calculo la fft del ruido
RUIDO2=fft(ruido2)
RUIDO2abs=np.abs(RUIDO2)

plt.figure(3)
plt.plot(ff,20*np.log10(X2abs), label='X2(SNR=10) abs en V' )
plt.plot(ff,20*np.log10(S1abs), label='S1 abs en V' )
plt.plot(ff,20*np.log10(RUIDO2abs), label='RUIDO 2(SNR=10) abs en V' )
plt.title('FFT de la senoidal')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show ()

#se ve que el orden de magnitud más alto tapa al de abajo. 
#--------------------------------------------------------
#AHORA CALIBRAMOS
RUIDO2_calib=(1/N)*fft(ruido2)
# X1_calib=(1/N)*fft(x1)
X2_calib1=(1/N)*fft(x2)
S1_calib=(1/N)*fft(s1)
X2_calib=RUIDO2_calib+S1_calib

plt.figure(4)
plt.plot(ff,20*np.log10(np.abs(X2_calib)), label='X2(SNR=10) abs en V' )
plt.plot(ff, 20*np.log10(X2abs), label='X2 (SN10) sin calibrar')
plt.title('FFT de la senoidal casi calibrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show

#no esta en 0dB.
#-----------------------------------------------------
#ahora grafico el cuadrado del módulo de x, que reepresenta la tension V, aplicamos que 
#debería dar lo mismo que 10 log de la potencia

plt.figure(5)
plt.plot(ff,10*np.log10((2*np.abs(X2_calib))**2), label='X2 densidad espectral de potencia(SNR=10) abs en V' ) #densidad espectral de potencia
plt.title('FFT de la senoidal calibrada subiendo los 3dB')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show

#la potencia de la senoidal la tenemos repartida entre las dos deltas, una que esta en el ancho
#de banda digital entre 0 y nyquist y la otra entre nyquist y sampling
#como se ve en el gráfico faltan 3dB para llegar a los 0. Toda la potencia de 0 a nyquist. 
#por parseval, podes tomar de 0 a nyquist, podes sumar 3 dB, que es duplicar en veces 
#CUANDO HABLAMOS DE POTENCIA

#Aca entonces lo que se ve, es el valor del SNR más 10log(Bw), Bw=500Hz, Fs/2
 



# #-------------------------------------------------------
R = 200              # cantidad de realizaciones
fs = 1000            # frecuencia de muestreo

n = np.arange(0, N)            # vector de índices de tiempo -> (N,)
t = n / fs                     # vector de tiempo en segundos -> (N,)
fr = np.random.uniform(-2, 2, R)  # frecuencias aleatorias -> (R,)

# Reshape para que t sea columna (N,1) y fr sea fila (1,R)
t_col = t.reshape(-1, 1)       # (N,1)
fr_row = fr.reshape(1, -1)     # (1,R)

# Generar matrices con broadcasting, sin usar bucles
# t_col * fr_row -> matriz (N,R), cada columna tiene el tiempo, cada fila usa una freq
sen_matriz = amp * np.sin(2 * np.pi * fr_row * t_col)

print("t_col shape:", t_col.shape)      # (N,1)
print("fr_row shape:", fr_row.shape)    # (1,R)
print("sen_matriz shape:", sen_matriz.shape)  # (N,R)


#lo que quiere es generar a partir del vector de fr una matriz que tenga como fila
#el tiempo, o sea las muestras. Y como columna quiere las realizaciones, es decir las 
#diferentes senoidales con ruido

sen_matriz=amp*np.sin(2*np.pi*fs/4*t)

# a0=np.sqrt(2)
# Omega0=fs/4
# ps= a0**2/2


# Omega1 = Omega0 + fr[:, None] * (fs/ N)
# n = np.arange(N)
# x_gen=a0* np.sin(Omega1 * n)

# #X_gen=fft(x_gen)
# ff=np.arange(N)*df
# #plt.plot(ff,20*np.log10(np.abs(X_gen)))
# #plt.grid(True)
# #plt.show()


# x=x_gen[5]


# def compute_noise_variance_from_snr(Ps, snr_db):
#     Pn = Ps / (10**(snr_db/10))
#     return Pn

# Pn1=compute_noise_variance_from_snr(ps, SNR1)
# ruido1 = np.random.normal(0,np.sqrt(Pn1), N)
# print(f"potencia de SNR {Pn1:3.3f}")
# var_ruido=np.var(ruido1)
# print(f"potencia del ruido -> {var_ruido:3.3f}")

# Pn2=compute_noise_variance_from_snr(ps, SNR2)
# ruido2 = np.sqrt(Pn2) * np.random.normal(len(x))

# x_ruidosa1 = x + ruido1
# x_ruidosa2 = x + ruido2

# X_ruidosa1= fft(x_ruidosa1)
# Xabs_ruidosa1= np.abs(X_ruidosa1)

# plt.plot(ff,20*np.log10(Xabs_ruidosa1))
# plt.grid(True)
# plt.show()
# #-----------------------------------------------------
# # Ventanas a usar
# ventanas = {
#     "Rectangular": windows.boxcar(N),
#     "Hamming": windows.hamming(N),
#     "Hann": windows.hann(N),
#     "Blackman Harris": windows.blackmanharris(N),
#     "Flattop": windows.flattop(N)
# }

# # FFT para estimación con alta resolución
# NFFT = 9 * N   # zero-padding para mayor resolución en frecuencia
# #freqs = np.linspace(-np.pi, np.pi, NFFT, endpoint=False)  # eje en rad/muestra centrado

# plt.figure(figsize=(10, 5))

# for nombre, ventana in ventanas.items():
#     # FFT centrada
#     W = fft(ventana, NFFT)
#     W = fftshift(W)
#     W_db = 20 * np.log10(np.abs(W) / np.max(np.abs(W)) + 1e-12)

#     # eje en múltiplos de Δf
#    # k = np.arange(-NFFT//2, NFFT//2)  # enteros
#    # plt.plot(k, W_db, label=nombre)
    
    
# #----------------------------------------------
# #res esp  
# #plt.figure(figsize=(12, 6))

# for nombre, w in ventanas.items():
#     xw = x_ruidosa1 * w  # aplicar ventana
#     X = fft(xw, NFFT)
#     X = fftshift(X)
#     X_db = 20*np.log10(np.abs(X)/np.max(np.abs(X)) + 1e-12)
    
#     # eje en múltiplos de Δf
#     k = np.arange(-NFFT//2, NFFT//2)
#     plt.plot(k, X_db, label=nombre)

# plt.title("Estimación espectral con diferentes ventanas")
# plt.xlabel("Frecuencia [múltiplos de Δf]")
# plt.ylabel("Magnitud [dB]")
# plt.ylim([-80, 5])
# plt.xlim(-200, 500)  # zoom
# plt.grid(True)
# plt.legend()
# plt.show()