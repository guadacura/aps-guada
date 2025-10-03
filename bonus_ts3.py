# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:40:35 2025

@author: USUARIO
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Coeficientes de tus sistemas LTI
# Sistema 1: y[n] = 0.03*x[n] + 0.05*x[n-1] + 0.03*x[n-2] + 1.5*y[n-1] - 0.5*y[n-2]
b1 = [0.03, 0.05, 0.03]  # Coeficientes de x
a1 = [1, -1.5, 0.5]      # Coeficientes de y

# Sistema 2: y[n] = x[n] + 3*x[n-10]
b2 = [1] + [0]*9 + [3]   # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
a2 = [1]                 # Solo componente actual

# Sistema 3: y[n] = x[n] + 3*y[n-10]
b3 = [1]                 # Solo componente actual de x
a3 = [1] + [0]*9 + [-3]  # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3]

# Calculamos la respuesta en frecuencia
fm = 100000  # Frecuencia de muestreo
frecuencias = np.linspace(0, fm/2, 1000)  # De 0 a fm/2 (Nyquist)

# Usamos freqz para calcular la respuesta
w1, h1 = signal.freqz(b1, a1, worN=frecuencias, fs=fm)
w2, h2 = signal.freqz(b2, a2, worN=frecuencias, fs=fm)
w3, h3 = signal.freqz(b3, a3, worN=frecuencias, fs=fm)

# Crear figura con 6 subplots (3 filas, 2 columnas)
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# ===== SISTEMA 1 =====
# Magnitud del Sistema 1
axes[0, 0].plot(w1, 20 * np.log10(np.abs(h1)), 'b-', linewidth=2)
axes[0, 0].set_title('Sistema 1: Magnitud\n$y[n] = 0.03x[n] + 0.05x[n-1] + 0.03x[n-2] + 1.5y[n-1] - 0.5y[n-2]$')
axes[0, 0].set_xlabel('Frecuencia [Hz]')
axes[0, 0].set_ylabel('Magnitud [dB]')
axes[0, 0].grid(True)
axes[0, 0].set_xlim(0, fm/2)

# Fase del Sistema 1
axes[0, 1].plot(w1, np.angle(h1, deg=True), 'b-', linewidth=2)
axes[0, 1].set_title('Sistema 1: Fase\n$y[n] = 0.03x[n] + 0.05x[n-1] + 0.03x[n-2] + 1.5y[n-1] - 0.5y[n-2]$')
axes[0, 1].set_xlabel('Frecuencia [Hz]')
axes[0, 1].set_ylabel('Fase [grados]')
axes[0, 1].grid(True)
axes[0, 1].set_xlim(0, fm/2)

# ===== SISTEMA 2 =====
# Magnitud del Sistema 2
axes[1, 0].plot(w2, 20 * np.log10(np.abs(h2)), 'r-', linewidth=2)
axes[1, 0].set_title('Sistema 2: Magnitud\n$y[n] = x[n] + 3x[n-10]$')
axes[1, 0].set_xlabel('Frecuencia [Hz]')
axes[1, 0].set_ylabel('Magnitud [dB]')
axes[1, 0].grid(True)
axes[1, 0].set_xlim(0, fm/2)

# Fase del Sistema 2
axes[1, 1].plot(w2, np.angle(h2, deg=True), 'r-', linewidth=2)
axes[1, 1].set_title('Sistema 2: Fase\n$y[n] = x[n] + 3x[n-10]$')
axes[1, 1].set_xlabel('Frecuencia [Hz]')
axes[1, 1].set_ylabel('Fase [grados]')
axes[1, 1].grid(True)
axes[1, 1].set_xlim(0, fm/2)

# ===== SISTEMA 3 =====
# Magnitud del Sistema 3
axes[2, 0].plot(w3, 20 * np.log10(np.abs(h3)), 'g-', linewidth=2)
axes[2, 0].set_title('Sistema 3: Magnitud\n$y[n] = x[n] + 3y[n-10]$')
axes[2, 0].set_xlabel('Frecuencia [Hz]')
axes[2, 0].set_ylabel('Magnitud [dB]')
axes[2, 0].grid(True)
axes[2, 0].set_xlim(0, fm/2)

# Fase del Sistema 3
axes[2, 1].plot(w3, np.angle(h3, deg=True), 'g-', linewidth=2)
axes[2, 1].set_title('Sistema 3: Fase\n$y[n] = x[n] + 3y[n-10]$')
axes[2, 1].set_xlabel('Frecuencia [Hz]')
axes[2, 1].set_ylabel('Fase [grados]')
axes[2, 1].grid(True)
axes[2, 1].set_xlim(0, fm/2)

# Ajustar espaciado y mostrar
plt.tight_layout()
plt.show()

# También mostrar información adicional sobre cada sistema
print("=== ANÁLISIS DE LOS SISTEMAS ===")
print("\nSistema 1: Filtro IIR de 2do orden")
print("Coeficientes b (numerador):", b1)
print("Coeficientes a (denominador):", a1)
print("Este sistema tiene polos y ceros, es estable si los polos están dentro del círculo unitario")

print("\nSistema 2: Filtro FIR con retardo")
print("Coeficientes b:", b2)
print("Coeficientes a:", a2)
print("Sistema FIR - siempre estable, solo tiene ceros")

print("\nSistema 3: Filtro IIR con retardo recursivo")
print("Coeficientes b:", b3)
print("Coeficientes a:", a3)
print("Sistema IIR - puede ser inestable si |3| > 1 (que es el caso)")