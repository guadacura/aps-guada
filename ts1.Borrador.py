# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:28:00 2025

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

fs = 100000  # Frecuencia de muestreo
ts = 1/fs     # Tiempo entre muestras [s]
nn = 500      # Cantidad de muestras → duración total 5 ms
tt = np.arange(nn)*ts   # Vector de tiempo

# Función para graficar y mostrar ts, nn y energía
def grafico(x, y, titulo):
    E = np.sum(np.abs(y)**2) * ts  # Energía de la señal
    N = len(y)                     # Cantidad de muestras
    Ts = x[1]-x[0]                 # Tiempo entre muestras
    
    plt.figure(figsize=(8,4))
    line_hdls = plt.plot(x, y, 'g', label='Señal')
    plt.title(titulo)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    
    # Información en la leyenda
    info_text = f"ts = {Ts*1e6:.2f} µs, nn = {N}, Energía = {E:.4f} J"
    plt.legend(line_hdls, [info_text], loc='upper right')
    
    plt.tight_layout()
    plt.show()


# Señal 1: sinusoidal 2 kHz
ff_1 = 2000  # Hz
sen_1 = np.sin(2*np.pi*ff_1*tt)
grafico(tt, sen_1, 'Señal sinusoidal 2 kHz (S1)')

# Señal 2: amplificada y desfasada
sen_2 = 4*np.sin(2*np.pi*ff_1*tt + np.pi/2)
grafico(tt, sen_2, 'S1 amplificada y desfasada π/2')

# Señal 3: modulada en amplitud
ff_mod = 1000  # Hz
m = 0.7        # índice de modulación (0 < m <= 1)
s_moduladora = np.sin(2*np.pi*ff_mod*tt)
sen_3 = (1 + m*s_moduladora) * sen_1
env_sup = (1 + m*s_moduladora)
env_inf = -(1 + m*s_moduladora)

plt.figure(figsize=(8,4))
E = np.sum(np.abs(sen_3)**2) * ts  # Energía de la señal 3
N = len(sen_3)                     # Cantidad de muestras
Ts = tt[1]-tt[0] 

# Graficar
line1, = plt.plot(tt, sen_3, 'g', label='S1 modulada en amplitud')
line2, = plt.plot(tt, env_sup, 'r--', label='Envolvente superior')
line3, = plt.plot(tt, env_inf, 'y--', label='Envolvente inferior')

# Info extra
info_text = f"ts = {Ts*1e6:.2f} µs, nn = {N}, Energía = {E:.4f} J"

# Primera leyenda (las señales)
legend1 = plt.legend([line1, line2, line3],
                     [line1.get_label(), line2.get_label(), line3.get_label()],
                     loc='upper right')
plt.gca().add_artist(legend1)  # fijar esta leyenda

# Segunda leyenda (info aparte)
plt.legend([line1], [info_text], loc='lower right')

plt.title('Modulación')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.tight_layout()
plt.show()

def recorte_energia(signal, ts, porcentaje=0.75):
    
    # Energía original
    E_orig = np.sum(np.abs(signal)**2) * ts
    E_obj = porcentaje * E_orig

    # Máxima amplitud de la señal
    max_amp = np.max(np.abs(signal))
    umbral = max_amp

    # Búsqueda iterativa del umbral
    for thresh in np.linspace(max_amp, 0, 2000):
        rec = np.clip(signal, -thresh, thresh)
        E_rec = np.sum(np.abs(rec)**2) * ts
        if E_rec <= E_obj:
            umbral = thresh
            break

    signal_rec = np.clip(signal, -umbral, umbral)
    return signal_rec, umbral


def recorte_por_amplitud(x, factor=0.75):
    A = np.max(np.abs(x))
    u = factor * A
    return np.clip(x, -u, u), u

sen_4, umbral = recorte_energia(sen_3,ts,0.75)
sen_4_modif, umbral_modif = recorte_por_amplitud(sen_3,0.75)
grafico(tt, sen_4, 'Señal recortada en amplitud al 75% de la energía')
grafico(tt, sen_4_modif, 'Señal recortada al 75% de su amplitud')

# Señal 5: cuadrada 4 kHz
ff_2 = 4000  # Hz
sen_5 = signal.square(2*np.pi*ff_2*tt)
grafico(tt, sen_5, 'Señal cuadrada 4 kHz')

# Señal 6: pulso rectangular 10 ms

nn_pulso = int(15e-3*fs)   # número total de muestras dado 15 ms
tt_pulso = np.arange(nn_pulso)*ts   # vector de tiempo

sen_6 = np.zeros_like(tt_pulso)    # creo vector de igual longitud
n_pulso = int(10e-3*fs)            # duración del pulso = 10 ms
sen_6[:n_pulso] = 1                # vale 1 en esos 10 ms

grafico(tt_pulso, sen_6, 'Pulso rectangular 10 ms')

def prod_interno(s1, s2, ts, n1, n2, tol=1e-10):
    
    # Producto interno con np.dot
    p_int = np.dot(s1, s2) * ts

    if np.abs(p_int) < tol:
        print(f'Las señales {n1} y {n2} son ortogonales.')
    else:
        print(f'Las señales {n1} y {n2} NO son ortogonales.')


# Verificar ortogonalidad
prod_interno(sen_1, sen_2, ts, 'S1', 'S2', 1e-10)
prod_interno(sen_1, sen_3, ts, 'S1', 'S3', 1e-10)
prod_interno(sen_1, sen_4, ts, 'S1', 'S4', 1e-10)
prod_interno(sen_1, sen_4_modif, ts, 'S1', 'S4 modif', 1e-10)
prod_interno(sen_1, sen_5, ts, 'S1', 'S5', 1e-10)
prod_interno(sen_1, sen_6[:len(sen_1)], ts, 'S1', 'Pulso', 1e-10)

def correlacion(x, y, titulo):
    corr = np.correlate(x, y, mode='full') * ts
    lags = np.arange(-len(x)+1, len(x)) * ts
    
    plt.figure(figsize=(8,4))
    plt.plot(lags, corr, 'm')
    plt.title(titulo)
    plt.xlabel('Demora [s]')
    plt.ylabel('Correlación')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Autocorrelación de S1
correlacion(sen_1, sen_1, 'Autocorrelación de S1')

# Correlación entre S1 y las demás
correlacion(sen_1, sen_2, 'Correlación S1-S2')
correlacion(sen_1, sen_3, 'Correlación S1-S3')
correlacion(sen_1, sen_4, 'Correlación S1-S4')
correlacion(sen_1, sen_4_modif, 'Correlación S1-S4 modif')
correlacion(sen_1, sen_5, 'Correlación S1-S5')
correlacion(sen_1, sen_6[:len(sen_1)], 'Correlación S1-Pulso')

# Demostración:
# 2 sin(α) sin(β) = cos(α-β) - cos(α+β)

w = 2*np.pi*1000   # w = 2πf
alpha = w*tt       # α = ω·t
beta = (w/2)*tt    # β = ω/2·t   (la mitad de α)

lhs = 2*np.sin(alpha)*np.sin(beta)          # lado izquierdo
rhs = np.cos(alpha-beta) - np.cos(alpha+beta)  # lado derecho

plt.figure(figsize=(8,4))
plt.plot(tt, lhs, 'r', label='2 sin(α)·sin(β)')
plt.plot(tt, rhs, 'b--', label='cos(α-β) - cos(α+β)')
plt.legend()
plt.title('Verificación de la identidad trigonométrica')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()
