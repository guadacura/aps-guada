# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 11:59:19 2025

@author: USUARIO
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from numpy.fft import fft, fftshift, fftfreq
from scipy.optimize import curve_fit

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

pot_ruido1 = a0*2/(2*10*(SNR1/10))
pot_ruido2 = a0*2/(2*10*(SNR2/10))

#%% Matriz de senos
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

# Calcular FFTs
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

#%% Frecuencias para el eje
NFFT = N
freqs = fftfreq(NFFT*10, 1/fs)
# freqs_shifted = fftshift(freqs)

def graficar(X, nombre):
    #X_shifted = fftshift(X)
    X_db = 20 * np.log10(np.abs(X) / np.max(np.abs(X)) + 1e-12)
    
    k = np.linspace(0, N, len(X_db), endpoint=False)
    plt.plot(k, X_db, label=nombre)

# Grafico SNR = 3 dB
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
graficar(XX_sen_ruido1[:, 0], "Rectangular")
graficar(XX_r1_ham[:, 0], "Hamming")
graficar(XX_r1_bh[:, 0], "Blackman-Harris")
graficar(XX_r1_fl[:, 0], "Flattop")
plt.title("Estimación espectral con diferentes ventanas (SNR = 3 dB)")
plt.xlabel("Frecuencia [múltiplos de Δf]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
graficar(XX_sen_ruido2[:, 0], "Rectangular")
graficar(XX_r2_ham[:, 0], "Hamming")
graficar(XX_r2_bh[:, 0], "Blackman-Harris")
graficar(XX_r2_fl[:, 0], "Flattop")
plt.title("Estimación espectral con diferentes ventanas (SNR = 10 dB)")
plt.xlabel("Frecuencia [múltiplos de Δf]")
plt.ylabel("Magnitud [dB]")
plt.ylim([-80, 5])
plt.xlim(0, 1000)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%% Estimadores 
def estimar_amplitud_frecuencia(X_fft, frecuencias_fft):
    # Encontrar índices de frecuencias de 0 a Nyquist
    fs_nyquist = fs / 2
    idx_nyquist = np.where((frecuencias_fft >= 0) & (frecuencias_fft <= fs_nyquist))[0]
    frecuencias_nyquist = frecuencias_fft[idx_nyquist]
    
    # Extraer solo el espectro de 0 a Nyquist para todas las realizaciones
    X_nyquist = X_fft[idx_nyquist, :]
    
    # Estimador de amplitud en Omega0 para todas las realizaciones
    idx_omega0 = np.argmin(np.abs(frecuencias_nyquist - Omega0))
    amplitudes = np.abs(X_nyquist[idx_omega0, :])
    
    # Estimador de frecuencia: buscar máximo para cada realización
    idx_max = np.argmax(np.abs(X_nyquist), axis=0)
    frecuencias_estimadas = frecuencias_nyquist[idx_max]
    
    return amplitudes, frecuencias_estimadas

# Aplicar estimadores a todas las ventanas
frecuencias_verdaderas = Omega0 + fr * deltaF

# SNR = 3 dB
a1_r1_rect, omega1_r1_rect = estimar_amplitud_frecuencia(XX_sen_ruido1, freqs)
a1_r1_ham, omega1_r1_ham = estimar_amplitud_frecuencia(XX_r1_ham, freqs)
a1_r1_bh, omega1_r1_bh = estimar_amplitud_frecuencia(XX_r1_bh, freqs)
a1_r1_fl, omega1_r1_fl = estimar_amplitud_frecuencia(XX_r1_fl, freqs)

# SNR = 10 dB
a1_r2_rect, omega1_r2_rect = estimar_amplitud_frecuencia(XX_sen_ruido2, freqs)
a1_r2_ham, omega1_r2_ham = estimar_amplitud_frecuencia(XX_r2_ham, freqs)
a1_r2_bh, omega1_r2_bh = estimar_amplitud_frecuencia(XX_r2_bh, freqs)
a1_r2_fl, omega1_r2_fl = estimar_amplitud_frecuencia(XX_r2_fl, freqs)

# Graficamos histogramas
plt.figure(figsize=(14, 10))

# Amplitud SNR = 3 dB
plt.subplot(2, 2, 1)
plt.hist(a1_r1_rect,alpha=0.7, bins=20, label="Rectangular", density=True)
plt.hist(a1_r1_ham,alpha=0.7, bins=20, label="Hamming", density=True)
plt.hist(a1_r1_bh,alpha=0.7, bins=20, label="Blackman-Harris", density=True)
plt.hist(a1_r1_fl,alpha=0.7, bins=20, label="Flattop", density=True)
plt.axvline(a0, color='black', linestyle='--', linewidth=2, label=f'Verdadero: {a0:.3f}')
plt.title("Estimación de Amplitud (SNR = 3 dB)")
plt.xlabel("Amplitud Estimada")
plt.ylabel("Densidad de Probabilidad")
plt.grid(True)
plt.legend()

# Amplitud SNR = 10 dB
plt.subplot(2, 2, 2)
plt.hist(a1_r2_rect, alpha=0.7,bins=20, label="Rectangular", density=True)
plt.hist(a1_r2_ham,alpha=0.7, bins=20, label="Hamming", density=True)
plt.hist(a1_r2_bh,alpha=0.7, bins=20, label="Blackman-Harris", density=True)
plt.hist(a1_r2_fl,alpha=0.7, bins=20, label="Flattop", density=True)
plt.axvline(a0, color='black', linestyle='--', linewidth=2, label=f'Verdadero: {a0:.3f}')
plt.title("Estimación de Amplitud (SNR = 10 dB)")
plt.xlabel("Amplitud Estimada")
plt.ylabel("Densidad de Probabilidad")
plt.grid(True)
plt.legend()

# Histogramas de FRECUENCIA
# Frecuencia SNR = 3 dB - CON LÍMITES CORRECTOS
plt.subplot(2, 2, 3)
# Usar límites alrededor de 250 Hz para mejor visualización
# freq_min = Omega0 - 10  # 240 Hz
# freq_max = Omega0 + 10  # 260 Hz
bins_freq = 20

plt.hist(omega1_r1_rect, alpha=0.7, bins=20, label="Rectangular", density=True)
plt.hist(omega1_r1_ham, alpha=0.7, bins=bins_freq, label="Hamming", density=True)
plt.hist(omega1_r1_bh, alpha=0.7, bins=bins_freq, label="Blackman-Harris", density=True)
plt.hist(omega1_r1_fl, alpha=0.7, bins=bins_freq, label="Flattop", density=True)
plt.axvline(Omega0, color='black', linestyle='--', linewidth=2, label=f'Verdadero: {Omega0:.1f} Hz')
plt.title("Estimación de Frecuencia (SNR = 3 dB) - Zoom alrededor de 250 Hz")
plt.xlabel("Frecuencia Estimada [Hz]")
plt.ylabel("Densidad de Probabilidad")
plt.grid(True, alpha=0.3)
plt.legend()

# Frecuencia SNR = 10 dB - CON LÍMITES CORRECTOS
plt.subplot(2, 2, 4)
plt.hist(omega1_r2_rect, alpha=0.7, bins=bins_freq, label="Rectangular", density=True)
plt.hist(omega1_r2_ham, alpha=0.7, bins=bins_freq, label="Hamming", density=True)
plt.hist(omega1_r2_bh, alpha=0.7, bins=bins_freq, label="Blackman-Harris", density=True)
plt.hist(omega1_r2_fl, alpha=0.7, bins=bins_freq, label="Flattop", density=True)
plt.axvline(Omega0, color='black', linestyle='--', linewidth=2, label=f'Verdadero: {Omega0:.1f} Hz')
plt.title("Estimación de Frecuencia (SNR = 10 dB) - Zoom alrededor de 250 Hz")
plt.xlabel("Frecuencia Estimada [Hz]")
plt.ylabel("Densidad de Probabilidad")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

#%% Calcular estadísticas
def calcular_estadisticas(a1_estimado, omega1_estimado, a0_verdadera, omega_verdadero):
    # Estadísticas de amplitud
    mu_a1 = np.median(a1_estimado)
    sesgo_a1 = mu_a1 - a0_verdadera
    varianza_a1 = np.var(a1_estimado)
    
    # Estadísticas de frecuencia
    mu_omega1 = np.median(omega1_estimado)
    sesgo_omega1 = np.mean(mu_omega1 - omega_verdadero)
    varianza_omega1 = np.var(omega1_estimado)
    
    return {
        'amplitud': {'media': mu_a1, 'sesgo': sesgo_a1, 'varianza': varianza_a1},
        'frecuencia': {'media': mu_omega1, 'sesgo': sesgo_omega1, 'varianza': varianza_omega1}
    }

# Calcular estadísticas para SNR = 3 dB
estadisticas_ruido1 = {
    'Rectangular': calcular_estadisticas(a1_r1_rect, omega1_r1_rect, a0, frecuencias_verdaderas),
    'Hamming': calcular_estadisticas(a1_r1_ham, omega1_r1_ham, a0, frecuencias_verdaderas),
    'Blackman_harris': calcular_estadisticas(a1_r1_bh, omega1_r1_bh, a0, frecuencias_verdaderas),
    'Flattop': calcular_estadisticas(a1_r1_fl, omega1_r1_fl, a0, frecuencias_verdaderas)
}

# Calcular estadísticas para SNR = 10 dB
estadisticas_ruido2 = {
    'Rectangular': calcular_estadisticas(a1_r2_rect, omega1_r2_rect, a0, frecuencias_verdaderas),
    'Hamming': calcular_estadisticas(a1_r2_ham, omega1_r2_ham, a0, frecuencias_verdaderas),
    'Blackman_harris': calcular_estadisticas(a1_r2_bh, omega1_r2_bh, a0, frecuencias_verdaderas),
    'Flattop': calcular_estadisticas(a1_r2_fl, omega1_r2_fl, a0, frecuencias_verdaderas)
}

#%% Mostrar tablas
def mostrar_tabla(estadisticas, snr):
    print(f"\n{'='*80}")
    print(f"TABLA ESTADÍSTICA - SNR = {snr} dB")
    print(f"{'='*80}")
    
    print(f"\n{'ESTIMACIÓN DE AMPLITUD':^60}")
    print(f"{'-'*60}")
    print(f"{'Ventana':<15} {'Media':<12} {'Sesgo':<12} {'Varianza':<12}")
    print(f"{'-'*60}")
    
    for ventana, stats in estadisticas.items():
        amp = stats['amplitud']
        print(f"{ventana:<15} {amp['media']:>11.4f} {amp['sesgo']:>11.4f} {amp['varianza']:>11.4f}")
    
    print(f"\n{'ESTIMACIÓN DE FRECUENCIA':^60}")
    print(f"{'-'*60}")
    print(f"{'Ventana':<15} {'Media (Hz)':<12} {'Sesgo (Hz)':<12} {'Varianza':<12}")
    print(f"{'-'*60}")
    
    for ventana, stats in estadisticas.items():
        freq = stats['frecuencia']
        print(f"{ventana:<15} {freq['media']:>11.4f} {freq['sesgo']:>11.4f} {freq['varianza']:>11.4e}")

# Mostrar tablas
mostrar_tabla(estadisticas_ruido1, SNR1)
mostrar_tabla(estadisticas_ruido2, SNR2)

#%% ================== ZERO PADDING ==================
# Señal con ruido (SNR=10 dB para el ejemplo)
x_test = xx_sen_ruido2[:, 0]

# FFT sin zero-padding
X_nozp = np.abs(fft(x_test, n=N)) / N
freq_nozp = np.arange(N) * fs / N

# FFT con zero-padding (9*N ceros -> total 10*N)
X_zp = np.abs(fft(x_test, n=10*N)) / N
freq_zp = np.arange(10*N) * fs / (10*N)

# Graficar comparación 
plt.figure(figsize=(10,5))
plt.plot(freq_nozp[:N//2], 20*np.log10(X_nozp[:N//2]), label="Sin zero-padding")
plt.plot(freq_zp[:10*N//2], 20*np.log10(X_zp[:10*N//2]), label="Con zero-padding (9N)")
plt.axvline(Omega0, color="k", linestyle="--", label="Frecuencia verdadera")
plt.title("Efecto del zero-padding en el espectro (SNR=10 dB)")
plt.xlim(220,260)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.legend()
plt.grid(True)
plt.show()

# Búsqueda del máximo SOLO en frecuencias positivas (0 a Nyquist)
# Para sin zero-padding
freq_pos_nozp = freq_nozp[:N//2]  # Frecuencias de 0 a Nyquist
X_pos_nozp = X_nozp[:N//2]        # Magnitudes correspondientes

# Para con zero-padding
freq_pos_zp = freq_zp[:10*N//2]   # Frecuencias de 0 a Nyquist
X_pos_zp = X_zp[:10*N//2]         # Magnitudes correspondientes

# Encontrar el máximo solo en el rango positivo
idx_max_nozp = np.argmax(X_pos_nozp)
idx_max_zp = np.argmax(X_pos_zp)

freq_estimada_nozp = freq_pos_nozp[idx_max_nozp]
freq_estimada_zp = freq_pos_zp[idx_max_zp]

frecuencia_verdadera = Omega0 + fr[0] * deltaF  # Frecuencia verdadera para esta realización


print("="*60)
print("ANÁLISIS DEL EFECTO DEL ZERO-PADDING")
print("="*60)
print(f"Frecuencia verdadera: {frecuencia_verdadera:.4f} Hz")
print(f"Frecuencia estimada SIN zero-padding: {freq_estimada_nozp:.4f} Hz")
print(f"Frecuencia estimada CON zero-padding: {freq_estimada_zp:.4f} Hz")


#%% ================== ESTIMADORES ALTERNATIVOS ==================
def estimador_rms(x):
    """Estimador de amplitud en el dominio temporal usando RMS"""
    return np.sqrt(2) * np.sqrt(np.mean(x**2))

def estimador_qip(X, fs):
    """Estimador de frecuencia usando interpolación cuadrática del pico"""
    # Considerar solo frecuencias positivas (0 a Nyquist)
    n = len(X)
    X_pos = X[:n//2]  # Solo la mitad positiva del espectro
    
    k = np.argmax(np.abs(X_pos))
    if k == 0 or k == len(X_pos)-1:
        return k*fs/len(X)  # sin interpolación si es borde
    alpha = np.abs(X_pos[k-1])
    beta = np.abs(X_pos[k])
    gamma = np.abs(X_pos[k+1])
    p = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma)
    k_interp = k + p
    return k_interp*fs/len(X)

# Probar con una realización
x_example = xx_sen_ruido2[:, 0]

# Amplitud temporal
amp_rms = estimador_rms(x_example)

# Frecuencia espectral con interpolación (usando zero-padding para mejor resolución)
X_example = fft(x_example, n=10*N)
freq_qip = estimador_qip(X_example, fs)

# Frecuencia verdadera para esta realización
frecuencia_verdadera = Omega0 + fr[0] * deltaF

print("\n=== Estimadores alternativos ===")
print(f"Amplitud RMS estimada: {amp_rms:.3f} (valor verdadero: {a0})")
print(f"Frecuencia estimada (QIP): {freq_qip:.3f} Hz (valor verdadero: {frecuencia_verdadera:.4f} Hz")
print(f"Frecuencia verdadera (realización): {frecuencia_verdadera:.3f} Hz")
print(f"Error de frecuencia: {abs(freq_qip - frecuencia_verdadera):.3f} Hz")